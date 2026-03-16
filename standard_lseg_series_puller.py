from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import lseg.data as ld
import hashlib
import re


@dataclass
class SeriesPullConfig:
    field: str = "TR.PriceClose"
    output_col: str = "value"
    fallback_field: str | None = None
    intervals: tuple[str, ...] = ("daily",)
    asof_tolerance_days: int = 120
    max_retries: int = 3
    base_sleep_sec: float = 0.4
    batch_size: int = 10
    batch_pause_sec: float = 0.0
    min_asof_date: pd.Timestamp = pd.Timestamp("1900-01-01")
    force_refresh: bool = False
    cache_only: bool = False
    skip_known_bad_ids: bool = True
    bad_id_cooldown_days: int = 30
    series_specs: tuple["SeriesFieldSpec", ...] | None = None
    primary_output_col: str | None = None
    candidate_order: tuple[str, ...] = ("ISIN", "RIC_current", "RIC", "history")


@dataclass(frozen=True)
class SeriesFieldSpec:
    output_col: str
    fields: tuple[str, ...]
    intervals: tuple[str, ...] | None = None


def _clean_str(s: pd.Series) -> pd.Series:
    x = s.astype("string").str.strip()
    return x.where(x.notna() & (x != ""), pd.NA)


def _safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(text))


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


BAD_IDS_COLUMNS = ["firm_id", "last_failed_at", "reason", "n_candidates", "tried_ids"]


def _load_bad_ids_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=BAD_IDS_COLUMNS)
    try:
        d = pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=BAD_IDS_COLUMNS)
    for c in BAD_IDS_COLUMNS:
        if c not in d.columns:
            d[c] = pd.NA
    d = d[BAD_IDS_COLUMNS].copy()
    d["firm_id"] = d["firm_id"].astype("string").str.strip()
    d["last_failed_at"] = pd.to_datetime(d["last_failed_at"], errors="coerce").dt.normalize()
    d = d.dropna(subset=["firm_id", "last_failed_at"]).copy()
    d = d.sort_values(["firm_id", "last_failed_at"]).drop_duplicates(subset=["firm_id"], keep="last")
    return d.reset_index(drop=True)


def load_bad_firm_ids(path: Path, cooldown_days: int = 30) -> set[str]:
    d = _load_bad_ids_table(path)
    if d.empty:
        return set()
    if cooldown_days is not None and cooldown_days > 0:
        cutoff = pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=int(cooldown_days))
        d = d[d["last_failed_at"] >= cutoff].copy()
    return set(d["firm_id"].dropna().astype(str).tolist())


def append_bad_ids_rows(path: Path, rows: list[dict]) -> pd.DataFrame:
    old = _load_bad_ids_table(path)
    if rows:
        new = pd.DataFrame(rows)
        for c in BAD_IDS_COLUMNS:
            if c not in new.columns:
                new[c] = pd.NA
        new = new[BAD_IDS_COLUMNS].copy()
        new["firm_id"] = new["firm_id"].astype("string").str.strip()
        new["last_failed_at"] = pd.to_datetime(new["last_failed_at"], errors="coerce").dt.normalize()
        out = pd.DataFrame.from_records(
            old.to_dict("records") + new.to_dict("records"),
            columns=BAD_IDS_COLUMNS,
        )
    else:
        out = old.copy()
    out = out.dropna(subset=["firm_id", "last_failed_at"]).copy()
    out = out.sort_values(["firm_id", "last_failed_at"]).drop_duplicates(subset=["firm_id"], keep="last")
    _ensure_parent(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    out.to_csv(tmp, index=False)
    tmp.replace(path)
    return out.reset_index(drop=True)


def normalize_step_rows(df: pd.DataFrame, output_col: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["firm_id", "date", output_col, "rank", "id_type", "pull_id"])
    x = df.copy()
    x["date"] = pd.to_datetime(x.get("date"), errors="coerce").dt.normalize()
    if output_col not in x.columns:
        x[output_col] = np.nan
    x[output_col] = pd.to_numeric(x[output_col], errors="coerce")
    for c in ["rank", "id_type", "pull_id"]:
        if c not in x.columns:
            x[c] = pd.NA
    x = x[["firm_id", "date", output_col, "rank", "id_type", "pull_id"]]
    x = x.dropna(subset=["firm_id", "date"]).sort_values(["firm_id", "date"]).drop_duplicates(["firm_id", "date"], keep="last")
    return x.reset_index(drop=True)


def build_company_candidates(company_req: pd.DataFrame, candidate_order: tuple[str, ...] = ("ISIN", "RIC_current", "RIC", "history")) -> list[tuple[str, str]]:
    q = company_req.copy().sort_values("date")
    out: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def _add(id_type: str, value: object) -> None:
        if pd.isna(value):
            return
        v = str(value).strip()
        if not v:
            return
        if id_type == "ISIN" and v.upper().startswith("ISIN:"):
            v = v.split(":", 1)[1].strip()
        key = (id_type, v)
        if key in seen:
            return
        seen.add(key)
        out.append(key)

    source_map = {"ISIN": ("ISIN", "ISIN"), "RIC_current": ("RIC_current", "RIC"), "RIC": ("RIC", "RIC")}
    for src in candidate_order:
        if src == "history":
            continue
        if src not in source_map:
            continue
        col, id_type = source_map[src]
        if col not in q.columns:
            continue
        for v in q[col].dropna().astype(str):
            _add(id_type, v)

    if ("history" in candidate_order) and {"id_type", "pull_id"}.issubset(q.columns):
        for _, row in q[["id_type", "pull_id"]].dropna().iterrows():
            it = str(row["id_type"]).strip().upper()
            pid = str(row["pull_id"]).strip()
            if not pid:
                continue
            if it == "ISIN":
                _add("ISIN", pid)
            elif it == "RIC":
                _add("RIC", pid)

    return out


def extract_history_multi(raw: pd.DataFrame, specs: list[SeriesFieldSpec]) -> pd.DataFrame:
    output_cols = [sp.output_col for sp in specs]
    if raw is None or len(raw) == 0:
        return pd.DataFrame(columns=["date", *output_cols])

    # get_history often returns dates in the index; normalize by materializing the index.
    x = pd.DataFrame(raw).copy().reset_index()
    date_col = None
    for c in x.columns:
        if str(c).lower() in {"date", "timestamp"}:
            date_col = c
            break
    if date_col is None:
        date_col = x.columns[0]

    id_like = set()
    for c in x.columns:
        cl = str(c).lower()
        if cl in {"instrument", "ric", "isin"} or "instrument" in cl or cl.endswith("ric") or cl.endswith("isin"):
            id_like.add(c)

    value_cols = [c for c in x.columns if c != date_col and c not in id_like]

    def _norm(t: str) -> str:
        return str(t).upper().replace(" ", "")

    value_cols_norm = {c: _norm(c) for c in value_cols}

    out = pd.DataFrame({"date": pd.to_datetime(x[date_col], errors="coerce").dt.normalize()})
    out = out.dropna(subset=["date"]).sort_values("date").drop_duplicates(["date"], keep="last")

    for sp in specs:
        picked = None
        for f in sp.fields:
            fn = _norm(f)
            for c in value_cols:
                if fn in value_cols_norm[c]:
                    picked = c
                    break
            if picked is not None:
                break

        if picked is None and len(value_cols) == 1 and len(specs) == 1:
            picked = value_cols[0]

        if picked is None:
            out[sp.output_col] = np.nan
        else:
            tmp = pd.DataFrame(
                {
                    "date": pd.to_datetime(x[date_col], errors="coerce").dt.normalize(),
                    sp.output_col: pd.to_numeric(x[picked], errors="coerce"),
                }
            )
            tmp = tmp.dropna(subset=["date"]).sort_values("date").drop_duplicates(["date"], keep="last")
            out = out.merge(tmp, on="date", how="left")

    return out[["date", *output_cols]]


def map_history_to_asof_multi(req_dates: pd.Series, hist: pd.DataFrame, output_cols: list[str], tol_days: int) -> pd.DataFrame:
    left = pd.DataFrame({"date": pd.to_datetime(req_dates, errors="coerce").dt.normalize()}).dropna().sort_values("date")
    if left.empty:
        return pd.DataFrame(columns=["date", *output_cols])
    if hist is None or hist.empty:
        for c in output_cols:
            left[c] = np.nan
        return left

    keep_cols = ["date", *[c for c in output_cols if c in hist.columns]]
    right = hist[keep_cols].copy()
    right["date"] = pd.to_datetime(right["date"], errors="coerce").dt.normalize()
    for c in output_cols:
        if c not in right.columns:
            right[c] = np.nan
        right[c] = pd.to_numeric(right[c], errors="coerce")
    right = right.dropna(subset=["date"]).sort_values("date").drop_duplicates(["date"], keep="last")

    return pd.merge_asof(
        left,
        right[["date", *output_cols]],
        on="date",
        direction="backward",
        tolerance=pd.Timedelta(days=tol_days),
    )


class StandardSeriesPuller:
    def __init__(
        self,
        config: SeriesPullConfig,
        cache_dir: Path,
        step_rows_path: Path,
        checkpoint_path: Path,
        bad_ids_path: Path,
        bad_rows_log_path: Path | None = None,
    ) -> None:
        self.cfg = config
        self.series_specs = self._resolve_series_specs()
        self.output_cols = [sp.output_col for sp in self.series_specs]
        self.primary_col = self.cfg.primary_output_col or self.output_cols[0]
        if self.primary_col not in self.output_cols:
            raise ValueError(f"primary_output_col={self.primary_col} not in output_cols={self.output_cols}")
        self.cache_dir = cache_dir
        self.step_rows_path = step_rows_path
        self.checkpoint_path = checkpoint_path
        self.bad_ids_path = bad_ids_path
        self.bad_rows_log_path = bad_rows_log_path
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        _ensure_parent(self.step_rows_path)
        _ensure_parent(self.checkpoint_path)
        _ensure_parent(self.bad_ids_path)
        if self.bad_rows_log_path is not None:
            _ensure_parent(self.bad_rows_log_path)

    def _resolve_series_specs(self) -> list[SeriesFieldSpec]:
        if self.cfg.series_specs:
            specs = list(self.cfg.series_specs)
            if len(specs) == 0:
                raise ValueError("series_specs is empty.")
            return specs
        fields = [self.cfg.field]
        if self.cfg.fallback_field and self.cfg.fallback_field != self.cfg.field:
            fields.append(self.cfg.fallback_field)
        return [SeriesFieldSpec(output_col=self.cfg.output_col, fields=tuple(fields), intervals=None)]

    def _empty_hist(self) -> pd.DataFrame:
        return pd.DataFrame(columns=["date", *self.output_cols])

    def _normalize_step_rows_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=["firm_id", "date", *self.output_cols, "rank", "id_type", "pull_id"])
        x = df.copy()
        x["date"] = pd.to_datetime(x.get("date"), errors="coerce").dt.normalize()
        for c in self.output_cols:
            if c not in x.columns:
                x[c] = np.nan
            x[c] = pd.to_numeric(x[c], errors="coerce")
        for c in ["rank", "id_type", "pull_id"]:
            if c not in x.columns:
                x[c] = pd.NA
        x = x[["firm_id", "date", *self.output_cols, "rank", "id_type", "pull_id"]]
        x = x.dropna(subset=["firm_id", "date"]).sort_values(["firm_id", "date"]).drop_duplicates(["firm_id", "date"], keep="last")
        return x.reset_index(drop=True)

    def _cache_path_for_company_id(self, firm_id: str, id_type: str, pull_id: str) -> Path:
        base = _safe_name(firm_id).replace(".parquet", "")
        suffix = _safe_name(f"{id_type}_{pull_id}")
        return self.cache_dir / f"{base}__{suffix}.parquet"

    def _load_cache(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            return self._empty_hist()
        try:
            d = pd.read_parquet(path).copy()
        except Exception:
            return self._empty_hist()
        if "date" not in d.columns:
            return self._empty_hist()
        d["date"] = pd.to_datetime(d["date"], errors="coerce").dt.normalize()
        for c in self.output_cols:
            if c not in d.columns:
                d[c] = np.nan
            d[c] = pd.to_numeric(d[c], errors="coerce")
        d = d.dropna(subset=["date"]).sort_values("date").drop_duplicates(["date"], keep="last")
        return d[["date", *self.output_cols]]

    def _save_cache(self, path: Path, df: pd.DataFrame) -> None:
        d = df.copy()
        d["date"] = pd.to_datetime(d["date"], errors="coerce").dt.normalize()
        for c in self.output_cols:
            if c not in d.columns:
                d[c] = np.nan
            d[c] = pd.to_numeric(d[c], errors="coerce")
        d = d.dropna(subset=["date"]).sort_values("date").drop_duplicates(["date"], keep="last")
        _ensure_parent(path)
        tmp = path.with_suffix(path.suffix + ".tmp")
        d.to_parquet(tmp, index=False)
        tmp.replace(path)

    def _pull_segment(self, pull_id: str, start: pd.Timestamp, end: pd.Timestamp, id_type: str | None = None) -> tuple[pd.DataFrame, bool]:
        if pd.isna(start) or pd.isna(end) or start > end:
            return self._empty_hist(), False

        all_fields = []
        intervals = []
        for sp in self.series_specs:
            for f in sp.fields:
                if f not in all_fields:
                    all_fields.append(f)
            use_intervals = sp.intervals if sp.intervals else self.cfg.intervals
            for iv in use_intervals:
                if iv not in intervals:
                    intervals.append(iv)

        id_type_u = str(id_type or "").upper().strip()
        universe_variants = [str(pull_id).strip()]
        if id_type_u == "ISIN":
            base = str(pull_id).strip()
            plain = base.split(":", 1)[1].strip() if base.upper().startswith("ISIN:") else base
            universe_variants = [plain] if plain else []

        saw_unresolvable = False
        saw_non_error_response = False
        for universe_id in dict.fromkeys(universe_variants):
            unresolved_for_this_id = False
            out_acc = self._empty_hist()
            # Robust field fallback: if a combined request fails/returns empty, also try fields individually.
            field_plans = [all_fields]
            if len(all_fields) > 1:
                field_plans.extend([[f] for f in all_fields])

            for interval in intervals:
                if unresolved_for_this_id:
                    break
                for field_list in field_plans:
                    last_err = None
                    for r in range(self.cfg.max_retries):
                        try:
                            raw = ld.get_history(
                                universe=[universe_id],
                                fields=field_list,
                                start=start.strftime("%Y-%m-%d"),
                                end=end.strftime("%Y-%m-%d"),
                                interval=interval,
                            )
                            saw_non_error_response = True
                            out = extract_history_multi(raw, self.series_specs)
                            has_values = bool(out[self.output_cols].notna().any().any()) if (not out.empty) else False
                            if has_values:
                                frames = [x for x in [out_acc, out] if not x.empty]
                                if frames:
                                    out_acc = (
                                        pd.concat(frames, ignore_index=True)
                                        .sort_values("date")
                                        .drop_duplicates(["date"], keep="last")
                                        .reset_index(drop=True)
                                    )
                                else:
                                    out_acc = self._empty_hist()
                                # If we have at least one value for every output column, stop early.
                                if bool(out_acc[self.output_cols].notna().any().all()):
                                    return out_acc, False
                            break
                        except Exception as e:
                            last_err = e
                            msg = str(e)
                            if "Unable to resolve all requested identifiers" in msg:
                                saw_unresolvable = True
                                unresolved_for_this_id = True
                                break
                            time.sleep(self.cfg.base_sleep_sec * (2 ** r) + random.random() * 0.3)
                    if unresolved_for_this_id:
                        break
                    if last_err is not None:
                        continue
            if not out_acc.empty and bool(out_acc[self.output_cols].notna().any().any()):
                return out_acc, False
            if unresolved_for_this_id:
                continue

        if saw_unresolvable and (not saw_non_error_response):
            return self._empty_hist(), True
        return self._empty_hist(), False

    def _update_company_cache(
        self, firm_id: str, id_type: str, pull_id: str, start: pd.Timestamp, end: pd.Timestamp
    ) -> tuple[pd.DataFrame, bool]:
        path = self._cache_path_for_company_id(firm_id, id_type, pull_id)
        cached = self._empty_hist() if self.cfg.force_refresh else self._load_cache(path)

        has_values = bool(pd.to_numeric(cached.get(self.primary_col), errors="coerce").notna().any()) if not cached.empty else False
        if (
            (not cached.empty)
            and (not self.cfg.force_refresh)
            and has_values
            and cached["date"].min() <= start
            and cached["date"].max() >= end
        ):
            return cached, False
        if self.cfg.cache_only:
            return cached, False

        pulled, permanent_bad_id = self._pull_segment(pull_id=pull_id, start=start, end=end, id_type=id_type)
        frames = [x for x in [cached, pulled] if not x.empty]
        out = pd.concat(frames, ignore_index=True).sort_values("date").drop_duplicates(["date"], keep="last") if frames else self._empty_hist()
        if (not out.empty) or self.cfg.force_refresh:
            self._save_cache(path, out)
        return out, permanent_bad_id

    def run(self, request_rows: pd.DataFrame) -> dict[str, int]:
        req = request_rows.copy()
        for c in ["firm_id", "ISIN", "RIC_current", "RIC", "id_type", "pull_id"]:
            if c in req.columns:
                req[c] = _clean_str(req[c])

        req["date"] = pd.to_datetime(req["date"], errors="coerce").dt.normalize()
        req = req.dropna(subset=["firm_id", "date"]).drop_duplicates(["firm_id", "date"], keep="last")
        req = req[req["date"] >= self.cfg.min_asof_date].copy().reset_index(drop=True)
        if req.empty:
            raise ValueError("No valid request rows after cleaning.")

        company_candidates_map = {
            str(fid): build_company_candidates(g, candidate_order=self.cfg.candidate_order)
            for fid, g in req.groupby("firm_id", sort=False)
        }
        companies_all = req["firm_id"].dropna().astype(str).unique().tolist()
        companies_total = len(companies_all)

        existing_step_rows = self._normalize_step_rows_frame(pd.read_parquet(self.step_rows_path)) if self.step_rows_path.exists() else self._normalize_step_rows_frame(pd.DataFrame())

        req_cov = req[["firm_id", "date"]].copy()
        req_cov["firm_id"] = req_cov["firm_id"].astype(str)
        req_cov["date"] = pd.to_datetime(req_cov["date"], errors="coerce").dt.normalize()
        req_cov = req_cov.dropna(subset=["firm_id", "date"]).drop_duplicates(["firm_id", "date"], keep="last")

        step_cov = existing_step_rows[["firm_id", "date", self.primary_col]].copy()
        step_cov["firm_id"] = step_cov["firm_id"].astype(str)
        step_cov["date"] = pd.to_datetime(step_cov["date"], errors="coerce").dt.normalize()
        step_cov[self.primary_col] = pd.to_numeric(step_cov[self.primary_col], errors="coerce")
        step_cov = step_cov.dropna(subset=["firm_id", "date"]).drop_duplicates(["firm_id", "date"], keep="last")

        cov = req_cov.merge(step_cov, on=["firm_id", "date"], how="left")
        exp = cov.groupby("firm_id").size().rename("n_expected")
        got = cov.groupby("firm_id")[self.primary_col].apply(lambda s: int(s.notna().sum())).rename("n_found")
        cov_stats = pd.concat([exp, got], axis=1).fillna(0)
        cov_stats["n_expected"] = cov_stats["n_expected"].astype(int)
        cov_stats["n_found"] = cov_stats["n_found"].astype(int)

        full_coverage_firms = set(cov_stats.index[(cov_stats["n_expected"] > 0) & (cov_stats["n_found"] == cov_stats["n_expected"])].astype(str).tolist())
        partial_coverage_firms = set(cov_stats.index[(cov_stats["n_found"] > 0) & (cov_stats["n_found"] < cov_stats["n_expected"])].astype(str).tolist())
        no_coverage_firms = set(cov_stats.index[cov_stats["n_found"] == 0].astype(str).tolist())
        bad_hist_before = _load_bad_ids_table(self.bad_ids_path)
        bad_before_set = set(bad_hist_before["firm_id"].astype(str).tolist()) if not bad_hist_before.empty else set()
        known_bad_recent = (
            load_bad_firm_ids(self.bad_ids_path, cooldown_days=self.cfg.bad_id_cooldown_days)
            if (self.cfg.skip_known_bad_ids and (not self.cfg.force_refresh))
            else set()
        )
        bad_ids_firms = set([f for f in no_coverage_firms if f in known_bad_recent])
        companies = sorted(no_coverage_firms - bad_ids_firms)
        # Base resume set for checkpoint accounting.
        processed_companies = set(full_coverage_firms) | set(partial_coverage_firms) | set(bad_ids_firms)
        pre_bad_skip = len(bad_ids_firms)
        full_coverage = int(len(full_coverage_firms))
        partial_coverage = int(len(partial_coverage_firms))
        bad_ids_count = int(len(bad_ids_firms))
        remaining_expected = int(companies_total - full_coverage - partial_coverage - bad_ids_count)
        if remaining_expected != len(companies):
            remaining_expected = int(len(companies))

        print("\n" + "=" * 88)
        print("Standard Series Pull Overview")
        print("=" * 88)
        print("series_specs:", ", ".join([f"{sp.output_col}<-{list(sp.fields)}" for sp in self.series_specs]))
        print(f"request_rows: {len(req):,}")
        print(
            f"coverage: all_companies={companies_total:,} | full_coverage={full_coverage:,} | "
            f"partial_coverage={partial_coverage:,} | bad_ids={bad_ids_count:,} | remaining={remaining_expected:,}"
        )
        print(f"mode: {'CACHE_ONLY' if self.cfg.cache_only else 'CACHE+NETWORK'} | batch_size: {self.cfg.batch_size}")
        print("=" * 88)

        total_cand_calls = 0
        total_resolved = 0
        total_unresolved = 0
        total_bad_id_skips = int(pre_bad_skip)
        bad_rows: list[dict] = []
        n_batches = int(np.ceil(len(companies) / self.cfg.batch_size)) if companies else 0
        new_rows_out: list[dict] = []

        if not self.cfg.cache_only:
            ld.open_session()
        try:
            for b_ix, b_start in enumerate(range(0, len(companies), self.cfg.batch_size), start=1):
                b_end = min(len(companies), b_start + self.cfg.batch_size)
                batch_companies = companies[b_start:b_end]
                batch_new_rows = []
                batch_processed = []
                batch_bad_rows = []
                print(f"[BATCH {b_ix}/{n_batches}] companies={len(batch_companies)} idx={b_start+1}-{b_end}")

                for k, firm_id in enumerate(batch_companies, start=1):
                    company_req = req[req["firm_id"] == firm_id].copy().sort_values("date")
                    req_dates = company_req["date"].dropna().drop_duplicates().sort_values().reset_index(drop=True)
                    if req_dates.empty:
                        continue
                    start = pd.to_datetime(req_dates.min()).normalize()
                    end = pd.to_datetime(req_dates.max()).normalize()

                    panel = pd.DataFrame({"date": req_dates})
                    for c in self.output_cols:
                        panel[c] = np.nan
                    panel["rank"] = pd.NA
                    panel["id_type"] = pd.NA
                    panel["pull_id"] = pd.NA

                    attempted_ids: list[str] = []
                    cands = company_candidates_map.get(str(firm_id), [])
                    cand_used = 0

                    for rank, (cand_type, cand_id) in enumerate(cands, start=1):
                        if panel[self.output_cols].notna().all().all():
                            break
                        cand_type = str(cand_type).upper().strip()
                        cand_id = str(cand_id).strip()
                        if not cand_type or not cand_id:
                            continue

                        cand_used += 1
                        total_cand_calls += 1
                        attempted_ids.append(f"{cand_type}:{cand_id}")

                        hist, permanent_bad_id = self._update_company_cache(
                            firm_id=str(firm_id),
                            id_type=cand_type,
                            pull_id=cand_id,
                            start=start,
                            end=end,
                        )
                        if permanent_bad_id:
                            continue

                        mapped = map_history_to_asof_multi(req_dates, hist, output_cols=self.output_cols, tol_days=self.cfg.asof_tolerance_days)
                        panel = panel.merge(mapped.rename(columns={c: f"{c}_cand" for c in self.output_cols}), on="date", how="left")
                        any_filled = False
                        for c in self.output_cols:
                            fill_mask = panel[c].isna() & panel[f"{c}_cand"].notna()
                            if bool(fill_mask.any()):
                                any_filled = True
                            panel.loc[fill_mask, c] = panel.loc[fill_mask, f"{c}_cand"]
                        if any_filled:
                            panel.loc[panel["rank"].isna(), "rank"] = rank
                            panel.loc[panel["id_type"].isna(), "id_type"] = cand_type
                            panel.loc[panel["pull_id"].isna(), "pull_id"] = cand_id
                        panel = panel.drop(columns=[f"{c}_cand" for c in self.output_cols], errors="ignore")

                    panel["firm_id"] = str(firm_id)
                    panel = panel[["firm_id", "date", *self.output_cols, "rank", "id_type", "pull_id"]]

                    resolved = int(panel[self.primary_col].notna().sum())
                    unresolved = int(panel[self.primary_col].isna().sum())
                    total_resolved += resolved
                    total_unresolved += unresolved

                    if unresolved > 0:
                        miss = panel[panel[self.primary_col].isna()][["firm_id", "date"]].copy()
                        miss["reason"] = "no_data_after_fallback"
                        miss["n_candidates"] = len(cands)
                        miss["tried_ids"] = "|".join(dict.fromkeys(attempted_ids))
                        miss_rows = miss.to_dict("records")
                        bad_rows.extend(miss_rows)
                        batch_bad_rows.extend(miss_rows)

                    batch_new_rows.extend(panel.to_dict("records"))
                    batch_processed.append(str(firm_id))

                    resolved_dates = pd.to_datetime(panel.loc[panel[self.primary_col].notna(), "date"], errors="coerce").dropna()
                    pulled_range = "NA:NA" if resolved_dates.empty else f"{resolved_dates.min().date()}:{resolved_dates.max().date()}"
                    range_in_index = f"{start.date()}:{end.date()}"
                    tried_preview = " | ".join(dict.fromkeys(attempted_ids)) if attempted_ids else "NA"
                    print(
                        f"[BATCH {b_ix}/{n_batches}] [{b_start+k}/{len(companies)}] "
                        f"firm_id={firm_id} | cand_used={cand_used}/{len(cands)} | "
                        f"unresolved={unresolved} | found_{self.primary_col}={resolved} | "
                        f"range_in_index={range_in_index} | pulled_range={pulled_range} | tried_ids: {tried_preview}"
                    )

                if batch_new_rows:
                    batch_df = self._normalize_step_rows_frame(pd.DataFrame(batch_new_rows))
                    prev = self._normalize_step_rows_frame(pd.read_parquet(self.step_rows_path)) if self.step_rows_path.exists() else pd.DataFrame(columns=batch_df.columns)
                    frames = [f for f in [prev, batch_df] if (f is not None and not f.empty)]
                    if not frames:
                        combined = pd.DataFrame(columns=batch_df.columns)
                    elif len(frames) == 1:
                        combined = frames[0].copy()
                    else:
                        combined = pd.concat(frames, ignore_index=True)
                    combined = combined.sort_values(["firm_id", "date"]).drop_duplicates(["firm_id", "date"], keep="last")
                    combined.to_parquet(self.step_rows_path, index=False)
                    new_rows_out = combined.to_dict("records")

                if batch_processed:
                    processed_companies.update(batch_processed)
                    ckpt_payload = {
                        "processed_companies": sorted(processed_companies),
                        "remaining_companies": max(0, companies_total - len(processed_companies)),
                        "updated_at_utc": pd.Timestamp.utcnow().isoformat(),
                        "rows": int(len(new_rows_out)),
                    }
                    self.checkpoint_path.write_text(json.dumps(ckpt_payload, ensure_ascii=False, indent=2))

                # Persist bad_ids incrementally per batch so progress survives interruptions.
                if batch_bad_rows:
                    batch_bad_df = pd.DataFrame(batch_bad_rows)
                    batch_bad_id_rows = []
                    for fid, grp in batch_bad_df.groupby("firm_id"):
                        tried = grp["tried_ids"].dropna().astype(str).iloc[0] if ("tried_ids" in grp.columns and grp["tried_ids"].notna().any()) else pd.NA
                        batch_bad_id_rows.append(
                            {
                                "firm_id": str(fid),
                                "last_failed_at": pd.Timestamp.utcnow().normalize(),
                                "reason": "no_data_all_candidates",
                                "n_candidates": int(pd.to_numeric(grp.get("n_candidates"), errors="coerce").max()) if "n_candidates" in grp.columns else pd.NA,
                                "tried_ids": tried,
                            }
                        )
                    bad_hist_batch = append_bad_ids_rows(self.bad_ids_path, batch_bad_id_rows)

                if self.cfg.batch_pause_sec > 0 and b_ix < n_batches:
                    time.sleep(self.cfg.batch_pause_sec)
        finally:
            if not self.cfg.cache_only:
                try:
                    ld.close_session()
                except Exception:
                    pass

        if bad_rows and (self.bad_rows_log_path is not None):
            bad_df = pd.DataFrame(bad_rows)
            if self.bad_rows_log_path.exists():
                old = pd.read_csv(self.bad_rows_log_path)
                out = pd.concat([old, bad_df], ignore_index=True)
            else:
                out = bad_df
            out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
            out = out.drop_duplicates(subset=["firm_id", "date", "reason"], keep="last")
            out.to_csv(self.bad_rows_log_path, index=False)

        bad_id_rows = []
        if bad_rows:
            bad_rows_df = pd.DataFrame(bad_rows)
            for fid, grp in bad_rows_df.groupby("firm_id"):
                tried = grp["tried_ids"].dropna().astype(str).iloc[0] if ("tried_ids" in grp.columns and grp["tried_ids"].notna().any()) else pd.NA
                bad_id_rows.append(
                    {
                        "firm_id": str(fid),
                        "last_failed_at": pd.Timestamp.utcnow().normalize(),
                        "reason": "no_data_all_candidates",
                        "n_candidates": int(pd.to_numeric(grp.get("n_candidates"), errors="coerce").max()) if "n_candidates" in grp.columns else pd.NA,
                        "tried_ids": tried,
                    }
                )
        bad_hist_after = append_bad_ids_rows(self.bad_ids_path, bad_id_rows)
        total_bad_ids_added = len(set(r["firm_id"] for r in bad_id_rows) - bad_before_set) if bad_id_rows else 0

        print(
            f"Done: companies_total={companies_total}, run_remaining_start={len(companies)}, candidate_calls={total_cand_calls}, "
            f"resolved_rows={total_resolved}, unresolved_rows={total_unresolved}, found_{self.primary_col}={total_resolved}, "
            f"full_coverage={full_coverage}, partial_coverage={partial_coverage}, bad_ids={bad_ids_count}, "
            f"bad_id_skip={total_bad_id_skips}, bad_ids_added={total_bad_ids_added}, known_bad_ids_now={len(bad_hist_after)}"
        )

        return {
            "companies_total": companies_total,
            "full_coverage": full_coverage,
            "partial_coverage": partial_coverage,
            "bad_ids": bad_ids_count,
            "remaining": int(remaining_expected),
            "run_remaining_start": len(companies),
            "candidate_calls": total_cand_calls,
            "resolved_rows": total_resolved,
            "unresolved_rows": total_unresolved,
            "bad_id_skip": total_bad_id_skips,
            "bad_ids_added": total_bad_ids_added,
            "known_bad_ids_now": int(len(bad_hist_after)),
        }

    def load_step_rows(self) -> pd.DataFrame:
        if self.step_rows_path.exists():
            return self._normalize_step_rows_frame(pd.read_parquet(self.step_rows_path))
        return self._normalize_step_rows_frame(pd.DataFrame())


def build_request_rows(source_df: pd.DataFrame, value_cols: list[str] | None = None) -> pd.DataFrame:
    value_cols = value_cols or []
    req_cols = [c for c in ["firm_id", "date", "ISIN", "RIC_current", "RIC", "id_type", "pull_id", *value_cols] if c in source_df.columns]
    if "firm_id" not in req_cols or "date" not in req_cols:
        raise ValueError("source_df must contain at least firm_id and date.")
    req = source_df[req_cols].copy()
    req["date"] = pd.to_datetime(req["date"], errors="coerce").dt.normalize()
    req = req.dropna(subset=["firm_id", "date"]).drop_duplicates(["firm_id", "date"], keep="last")
    return req.reset_index(drop=True)


def build_ltg_request_rows(source_df: pd.DataFrame, output_col: str = "LTG") -> pd.DataFrame:
    return build_request_rows(source_df=source_df, value_cols=[output_col])


def prepare_series_source_df(source_df: pd.DataFrame) -> pd.DataFrame:
    df = source_df.copy()
    if "date" not in df.columns:
        raise ValueError("source_df must contain 'date'.")
    if "firm_id" not in df.columns:
        raise ValueError("source_df must contain 'firm_id'.")

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    for c in ["firm_id", "ISIN", "RIC_current", "RIC", "id_type", "pull_id"]:
        if c in df.columns:
            df[c] = _clean_str(df[c])

    if "id_type" not in df.columns:
        df["id_type"] = np.select(
            [
                df.get("ISIN", pd.Series(pd.NA, index=df.index)).notna(),
                df.get("RIC_current", pd.Series(pd.NA, index=df.index)).notna(),
                df.get("RIC", pd.Series(pd.NA, index=df.index)).notna(),
            ],
            ["ISIN", "RIC", "RIC"],
            default=pd.NA,
        )
    if "pull_id" not in df.columns:
        df["pull_id"] = np.select(
            [
                df.get("ISIN", pd.Series(pd.NA, index=df.index)).notna(),
                df.get("RIC_current", pd.Series(pd.NA, index=df.index)).notna(),
                df.get("RIC", pd.Series(pd.NA, index=df.index)).notna(),
            ],
            [
                df.get("ISIN", pd.Series(pd.NA, index=df.index)),
                df.get("RIC_current", pd.Series(pd.NA, index=df.index)),
                df.get("RIC", pd.Series(pd.NA, index=df.index)),
            ],
            default=pd.NA,
        )
    df["id_type"] = _clean_str(df["id_type"])
    df["pull_id"] = _clean_str(df["pull_id"])
    return df


def _run_series_internal(
    source_df: pd.DataFrame,
    config: SeriesPullConfig,
    cache_dir: Path,
    step_rows_path: Path,
    checkpoint_path: Path,
    bad_ids_path: Path,
    bad_rows_log_path: Path | None = None,
    skip_filled_primary: bool = False,
    merge_back: bool = True,
    diag_prefix: str = "",
) -> dict:
    src = prepare_series_source_df(source_df)
    value_cols = [sp.output_col for sp in (list(config.series_specs) if config.series_specs else [SeriesFieldSpec(config.output_col, (config.field,))])]
    req_all = build_request_rows(src, value_cols=value_cols)
    req_all = req_all[req_all["date"] >= config.min_asof_date].copy().reset_index(drop=True)
    primary_col = config.primary_output_col or value_cols[0]
    if skip_filled_primary and primary_col in req_all.columns:
        req = req_all[pd.to_numeric(req_all[primary_col], errors="coerce").isna()].copy()
    else:
        req = req_all.copy()

    puller = StandardSeriesPuller(
        config=config,
        cache_dir=cache_dir,
        step_rows_path=step_rows_path,
        checkpoint_path=checkpoint_path,
        bad_ids_path=bad_ids_path,
        bad_rows_log_path=bad_rows_log_path,
    )
    if req.empty:
        append_bad_ids_rows(bad_ids_path, [])
        empty_step = puller.load_step_rows()
        if not step_rows_path.exists():
            empty_step.to_parquet(step_rows_path, index=False)
        if not checkpoint_path.exists():
            checkpoint_path.write_text(
                json.dumps(
                    {
                        "processed_companies": [],
                        "remaining_companies": int(req_all["firm_id"].nunique()) if not req_all.empty else 0,
                        "updated_at_utc": pd.Timestamp.utcnow().isoformat(),
                        "rows": int(len(empty_step)),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        stats = {
            "companies_total": int(req_all["firm_id"].nunique()) if not req_all.empty else 0,
            "full_coverage": 0,
            "partial_coverage": 0,
            "bad_ids": 0,
            "remaining": 0,
            "run_remaining_start": 0,
            "candidate_calls": 0,
            "resolved_rows": 0,
            "unresolved_rows": 0,
            "bad_id_skip": 0,
            "bad_ids_added": 0,
            "known_bad_ids_now": len(_load_bad_ids_table(bad_ids_path)),
        }
    else:
        stats = puller.run(req)

    step_rows = puller.load_step_rows()
    merged = src.copy()
    if merge_back:
        add_cols = ["firm_id", "date", *value_cols, "rank", "id_type", "pull_id"]
        add = step_rows[add_cols].copy() if not step_rows.empty else pd.DataFrame(columns=add_cols)
        ren = {c: f"{diag_prefix}{c}" for c in ["rank", "id_type", "pull_id"]} if diag_prefix else {"rank": "rank", "id_type": "id_type", "pull_id": "pull_id"}
        add = add.rename(columns=ren)
        merged = merged.merge(add, on=["firm_id", "date"], how="left", suffixes=("", "_new"))
        for c in value_cols:
            new_c = f"{c}_new"
            if new_c in merged.columns:
                merged[c] = pd.to_numeric(merged[new_c], errors="coerce").combine_first(pd.to_numeric(merged[c], errors="coerce") if c in merged.columns else pd.Series(np.nan, index=merged.index))
                merged = merged.drop(columns=[new_c], errors="ignore")

    return {
        "stats": stats,
        "request_rows": req,
        "request_rows_all": req_all,
        "step_rows": step_rows,
        "merged_df": merged,
        "puller": puller,
    }


def to_quarter_end_dates(dates: Iterable[pd.Timestamp | str]) -> pd.Series:
    d = pd.to_datetime(pd.Series(list(dates)), errors="coerce")
    q = pd.PeriodIndex(d.dropna(), freq="Q").to_timestamp(how="end").normalize()
    return pd.Series(q).drop_duplicates().sort_values().reset_index(drop=True)


@dataclass
class DailyReturnsPullConfig:
    target_end_date: pd.Timestamp = pd.Timestamp("2025-12-31")
    force_refresh: bool = False
    skip_lseg_pull: bool = False
    precheck_tol_days: int = 5
    coverage_tol_days: int = 5
    checkpoint_every_n_pulls: int = 25
    batch_size: int = 10
    skip_known_bad_ids: bool = True
    bad_id_cooldown_days: int = 30
    pull_abort_on_rate_limit: bool = True
    debug_firm_ids: set[str] | None = None


def _dr_safe_name(firm_id: str) -> str:
    h = hashlib.sha1(str(firm_id).encode("utf-8")).hexdigest()[:12]
    clean = re.sub(r"[^A-Za-z0-9._-]", "_", str(firm_id))
    return f"{clean[:80]}__{h}.parquet"


def _dr_extract_single_series(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["date", "value"])
    w = raw.copy().reset_index()
    if w.empty:
        return pd.DataFrame(columns=["date", "value"])

    def _col_name(c) -> str:
        if isinstance(c, tuple):
            return " ".join([str(x) for x in c if x is not None]).strip().lower()
        return str(c).strip().lower()

    def _is_date_named(c) -> bool:
        n = _col_name(c)
        return ("date" in n) or ("time" in n) or (n == "index")

    preferred = [c for c in w.columns if _is_date_named(c)]
    candidates = preferred if preferred else [w.columns[0]]
    date_col = None
    best_date_non_na = -1
    for c in candidates:
        d = pd.to_datetime(w[c], errors="coerce")
        n = int(d.notna().sum())
        if n > best_date_non_na:
            best_date_non_na = n
            date_col = c
    if date_col is None or best_date_non_na <= 0:
        return pd.DataFrame(columns=["date", "value"])

    w = w.rename(columns={date_col: "date"})
    w["date"] = pd.to_datetime(w["date"], errors="coerce")
    w = w.dropna(subset=["date"]).copy()

    value_col = None
    best_non_na = -1
    for c in w.columns:
        if c == "date":
            continue
        s_num = pd.to_numeric(w[c], errors="coerce")
        n = int(s_num.notna().sum())
        if n > best_non_na:
            best_non_na = n
            value_col = c
    if value_col is None or best_non_na <= 0:
        return pd.DataFrame(columns=["date", "value"])

    out = w[["date", value_col]].copy().rename(columns={value_col: "value"})
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    return out.dropna(subset=["value"]).copy()


def _dr_values_to_returns(s: pd.Series, mode: str) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    if mode == "price_level":
        return x.pct_change()
    abs_q99 = np.nanpercentile(np.abs(x.dropna()), 99) if x.notna().any() else np.nan
    return x / 100.0 if np.isfinite(abs_q99) and abs_q99 > 1.5 else x


def _dr_is_rate_limit_message(msg: str) -> bool:
    m = str(msg).lower()
    return ("too many requests" in m) or ("rate limit" in m) or ("http 429" in m) or ("status 429" in m)


def _dr_is_unable_to_resolve_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return (
        ("unable to resolve all requested identifiers" in msg)
        or ("universe is not found" in msg)
        or ("the universe is not found" in msg)
    )


def _dr_cache_path(cache_dir: Path, firm_id: str) -> Path:
    return cache_dir / _dr_safe_name(firm_id)


def _dr_load_cache(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["date", "ret"])
    d = pd.read_parquet(path)
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d["ret"] = pd.to_numeric(d["ret"], errors="coerce")
    return d.dropna(subset=["date", "ret"]).sort_values("date").copy()


def _dr_save_cache(path: Path, d: pd.DataFrame) -> None:
    _ensure_parent(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    d.sort_values("date").drop_duplicates(subset=["date"], keep="last").to_parquet(tmp, index=False)
    tmp.replace(path)


def _dr_normalize_seed(seed_returns: pd.DataFrame | None, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if seed_returns is None or seed_returns.empty:
        return pd.DataFrame(columns=["date", "ret"])
    x = seed_returns.copy()
    x["date"] = pd.to_datetime(x["date"], errors="coerce")
    x["ret"] = pd.to_numeric(x["ret"], errors="coerce")
    x = x.dropna(subset=["date", "ret"]).copy()
    x = x[(x["date"] >= start) & (x["date"] <= end)].copy()
    x = x.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    return x[["date", "ret"]]


def _dr_pull_one_company_returns(
    pull_id: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    id_type: str | None = None,
    max_retries: int = 4,
    base_sleep: float = 0.7,
) -> tuple[pd.DataFrame, str | None, str | None]:
    id_type = (id_type or "").upper()
    if id_type == "ISIN":
        plans = [("TR.TotalReturn", "return_like"), ("TR.PriceClose", "price_level"), ("TRDPRC_1", "price_level")]
    else:
        plans = [
            ("TR.TotalReturn", "return_like"),
            ("PCTCHNG", "return_like"),
            ("TR.PriceClose", "price_level"),
            ("TRDPRC_1", "price_level"),
        ]

    for field, mode in plans:
        last_err = None
        for r in range(max_retries):
            try:
                raw = ld.get_history(
                    universe=[pull_id],
                    fields=[field],
                    start=start.strftime("%Y-%m-%d"),
                    end=end.strftime("%Y-%m-%d"),
                    interval="daily",
                )
                ser = _dr_extract_single_series(raw)
                if ser.empty:
                    break
                ser = ser.sort_values("date")
                ser["ret"] = _dr_values_to_returns(ser["value"], mode=mode)
                ser = ser.dropna(subset=["ret"])[["date", "ret"]].copy()
                if not ser.empty:
                    return ser, field, mode
                break
            except Exception as e:
                last_err = e
                time.sleep(base_sleep * (2**r) + random.random() * 0.3)

        if last_err is not None:
            if _dr_is_rate_limit_message(str(last_err)):
                raise RuntimeError(f"RATE_LIMIT: {last_err}")

    return pd.DataFrame(columns=["date", "ret"]), None, None


def _dr_update_company_cache(
    cache_dir: Path,
    firm_id: str,
    pull_id: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    id_type: str | None,
    force_refresh: bool,
    seed_returns: pd.DataFrame | None,
) -> tuple[pd.DataFrame, str | None, str | None]:
    path = _dr_cache_path(cache_dir, firm_id)
    cached = pd.DataFrame(columns=["date", "ret"]) if force_refresh else _dr_load_cache(path)
    seed = _dr_normalize_seed(seed_returns, start=start, end=end)
    if not seed.empty:
        frames = [x for x in [cached, seed] if not x.empty]
        cached = pd.concat(frames, ignore_index=True).sort_values("date").drop_duplicates(subset=["date"], keep="last") if frames else pd.DataFrame(columns=["date", "ret"])

    segments: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    if cached.empty:
        segments.append((start, end))
    else:
        cmin, cmax = cached["date"].min(), cached["date"].max()
        if start < cmin:
            segments.append((start, cmin - pd.Timedelta(days=1)))
        if end > cmax:
            segments.append((cmax + pd.Timedelta(days=1), end))

    pulled_parts = []
    field_used = None
    mode_used = None
    for s, e in segments:
        if s > e:
            continue
        part, field_u, mode_u = _dr_pull_one_company_returns(pull_id=pull_id, start=s, end=e, id_type=id_type)
        if not part.empty:
            pulled_parts.append(part)
        field_used = field_used or field_u
        mode_used = mode_used or mode_u

    all_df = pd.concat([x for x in [cached] + pulled_parts if not x.empty], ignore_index=True) if pulled_parts else cached.copy()
    all_df = all_df.dropna(subset=["date", "ret"]).sort_values("date").drop_duplicates(subset=["date"], keep="last")
    if not all_df.empty or force_refresh:
        _dr_save_cache(path, all_df)
    return all_df, field_used, mode_used


def _run_daily_returns_internal(
    company_pull_map: pd.DataFrame,
    legacy_by_id: dict[tuple[str, str], pd.DataFrame],
    cache_dir: Path,
    manifest_path: Path,
    output_returns_all: Path,
    output_missing: Path,
    bad_ids_path: Path,
    step_rows_path: Path,
    step_ckpt_path: Path,
    config: DailyReturnsPullConfig | None = None,
) -> dict[str, int]:
    cfg = config or DailyReturnsPullConfig()
    _ensure_parent(manifest_path)
    _ensure_parent(output_returns_all)
    _ensure_parent(output_missing)
    _ensure_parent(bad_ids_path)
    _ensure_parent(step_rows_path)
    _ensure_parent(step_ckpt_path)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cpm = company_pull_map.copy()
    cpm["firm_id"] = cpm["firm_id"].astype("string")
    cpm["start_date"] = pd.to_datetime(cpm["start_date"], errors="coerce")
    cpm["end_date"] = pd.to_datetime(cpm["end_date"], errors="coerce")
    cpm = cpm.dropna(subset=["firm_id", "start_date", "end_date"]).copy()
    cpm["end_date"] = cpm["end_date"].clip(upper=cfg.target_end_date)

    initial_cache_files = {f.name for f in cache_dir.glob("*.parquet")}
    bad_hist_before = _load_bad_ids_table(bad_ids_path)
    bad_known_firms = (
        load_bad_firm_ids(bad_ids_path, cooldown_days=cfg.bad_id_cooldown_days)
        if (cfg.skip_known_bad_ids and not cfg.force_refresh)
        else set()
    )
    step_rows: list[dict] = []
    manifest_rows: list[dict] = []
    bad_rows: list[dict] = []
    all_company_returns: list[pd.DataFrame] = []

    # Coverage buckets for standardized overview and pull queue selection.
    def _cov_status(fid: str, start: pd.Timestamp, end: pd.Timestamp) -> str:
        d = _dr_load_cache(_dr_cache_path(cache_dir, fid))
        if d.empty:
            return "no_coverage"
        cmin = pd.to_datetime(d["date"], errors="coerce").min()
        cmax = pd.to_datetime(d["date"], errors="coerce").max()
        if pd.notna(cmin) and pd.notna(cmax) and cmin <= start and cmax >= end:
            return "full_coverage"
        return "partial_coverage"

    cov_rows = []
    for _, r in cpm.iterrows():
        fid = str(r["firm_id"])
        st = pd.to_datetime(r["start_date"]).normalize()
        en = pd.to_datetime(r["end_date"]).normalize()
        cov_rows.append((fid, _cov_status(fid, st, en)))
    cov_df = pd.DataFrame(cov_rows, columns=["firm_id", "cov_status"])

    all_companies = set(cov_df["firm_id"].astype(str).tolist())
    full_coverage_firms = set(cov_df.loc[cov_df["cov_status"] == "full_coverage", "firm_id"].astype(str).tolist())
    partial_coverage_firms = set(cov_df.loc[cov_df["cov_status"] == "partial_coverage", "firm_id"].astype(str).tolist())
    no_coverage_firms = set(cov_df.loc[cov_df["cov_status"] == "no_coverage", "firm_id"].astype(str).tolist())
    bad_ids_firms = set([f for f in no_coverage_firms if f in bad_known_firms])
    remaining_firms = sorted(no_coverage_firms - bad_ids_firms)
    processed_companies = set(full_coverage_firms) | set(partial_coverage_firms) | set(bad_ids_firms)

    # Always include already cached company returns in the consolidated output,
    # even when there is no remaining company to pull from LSEG.
    for _, row in cpm.iterrows():
        firm_id = str(row["firm_id"])
        if firm_id not in full_coverage_firms and firm_id not in partial_coverage_firms:
            continue

        start = pd.to_datetime(row["start_date"]).normalize()
        end = min(pd.to_datetime(row["end_date"]).normalize(), cfg.target_end_date)
        cached = _dr_load_cache(_dr_cache_path(cache_dir, firm_id))
        if cached.empty:
            continue

        cached = cached[(cached["date"] >= start) & (cached["date"] <= end)].copy()
        if cached.empty:
            continue

        cands = row.get("id_candidates", [])
        selected_id_type = pd.NA
        selected_pull_id = pd.NA
        if isinstance(cands, (list, tuple)) and len(cands) > 0:
            first = cands[0]
            if isinstance(first, (list, tuple)) and len(first) >= 2:
                selected_id_type = str(first[0]).upper().strip()
                selected_pull_id = str(first[1]).strip()

        expected = len(pd.bdate_range(start, end, freq="B"))
        cov_pct = 0.0
        if expected > 0:
            cov_pct = round(100.0 * cached["date"].dt.normalize().nunique() / expected, 2)

        tmp = cached.copy()
        tmp["firm_id"] = firm_id
        tmp["name"] = row.get("company_name", pd.NA)
        tmp["pull_id"] = selected_pull_id
        tmp["id_type"] = selected_id_type
        all_company_returns.append(tmp)

        mrow = {
            "firm_id": firm_id,
            "pull_id": selected_pull_id,
            "id_type": selected_id_type,
            "n_rows": int(len(cached)),
            "date_min": cached["date"].min(),
            "date_max": cached["date"].max(),
            "coverage_rate_pct": cov_pct,
            "field_used": pd.NA,
            "mode_used": pd.NA,
            "cache_path": str(_dr_cache_path(cache_dir, firm_id)),
            "status": "cached",
        }
        manifest_rows.append(mrow)
        step_rows.append(mrow)

    companies_total = len(all_companies)
    full_coverage = len(full_coverage_firms)
    partial_coverage = len(partial_coverage_firms)
    bad_ids_count = len(bad_ids_firms)
    remaining = companies_total - full_coverage - partial_coverage - bad_ids_count
    if remaining != len(remaining_firms):
        remaining = len(remaining_firms)

    print("\n" + "=" * 88)
    print("Standard Series Pull Overview")
    print("=" * 88)
    print("series_specs: ret<-['TR.TotalReturn','PCTCHNG','TR.PriceClose','TRDPRC_1']")
    print(f"request_rows: {len(cpm):,}")
    print(
        f"coverage: all_companies={companies_total:,} | full_coverage={full_coverage:,} | "
        f"partial_coverage={partial_coverage:,} | bad_ids={bad_ids_count:,} | remaining={remaining:,}"
    )
    print(f"mode: {'CACHE_ONLY' if cfg.skip_lseg_pull else 'CACHE+NETWORK'} | batch_size: {cfg.batch_size}")
    print("=" * 88)

    if not cfg.skip_lseg_pull:
        ld.open_session()
    try:
        pull_rows = cpm[cpm["firm_id"].astype("string").isin(remaining_firms)].copy()
        total = len(pull_rows)

        n_batches = int(np.ceil(total / cfg.batch_size)) if total > 0 else 0
        pull_idx = 0
        for b_ix, b_start in enumerate(range(0, total, cfg.batch_size), start=1):
            b_end = min(total, b_start + cfg.batch_size)
            batch_df = pull_rows.iloc[b_start:b_end]
            print(f"[BATCH {b_ix}/{n_batches}] companies={len(batch_df)} idx={b_start+1}-{b_end}")
            batch_rate_limit_abort = False
            batch_bad_rows: list[dict] = []
            batch_processed: list[str] = []

            for _, row in batch_df.iterrows():
                pull_idx += 1
                firm_id = str(row["firm_id"])
                start = pd.to_datetime(row["start_date"]).normalize()
                end = min(pd.to_datetime(row["end_date"]).normalize(), cfg.target_end_date)
                cands = row.get("id_candidates", [])
                if not isinstance(cands, (list, tuple)):
                    cands = []

                attempted_ids: list[str] = []
                final_data = pd.DataFrame(columns=["date", "ret"])
                selected_id_type = pd.NA
                selected_pull_id = pd.NA
                field_used = None
                mode_used = None
                hit_rate_limit = False
                for cand_type, cand_id in cands:
                    cand_type = str(cand_type).upper().strip()
                    cand_id = str(cand_id).strip()
                    if not cand_type or not cand_id:
                        continue
                    attempted_ids.append(f"{cand_type}:{cand_id}")
                    seed = legacy_by_id.get((cand_type, cand_id))
                    try:
                        data, f_used, m_used = _dr_update_company_cache(
                            cache_dir=cache_dir,
                            firm_id=firm_id,
                            pull_id=cand_id,
                            start=start,
                            end=end,
                            id_type=cand_type,
                            force_refresh=cfg.force_refresh,
                            seed_returns=seed,
                        )
                    except Exception as e:
                        if _dr_is_rate_limit_message(str(e)):
                            hit_rate_limit = True
                            if cfg.pull_abort_on_rate_limit:
                                break
                        continue

                    if not data.empty:
                        final_data = data
                        selected_id_type = cand_type
                        selected_pull_id = cand_id
                        field_used = f_used
                        mode_used = m_used
                        break

                if hit_rate_limit and cfg.pull_abort_on_rate_limit:
                    print("Pull loop stopped early due to rate limit. Remaining firms are left untouched for resume.")
                    batch_rate_limit_abort = True
                    break

                cov_pct = 0.0
                expected = len(pd.bdate_range(start, end, freq="B"))
                if expected > 0 and not final_data.empty:
                    cov_pct = round(100.0 * final_data["date"].dt.normalize().nunique() / expected, 2)

                index_range = f"{start.date()}:{end.date()}"
                if final_data.empty:
                    pulled_range = "NA:NA"
                else:
                    pulled_range = f"{final_data['date'].min().date()}:{final_data['date'].max().date()}"

                tried_preview = " | ".join(attempted_ids[:4]) if attempted_ids else "none"
                if len(attempted_ids) > 4:
                    tried_preview += f" | ... (+{len(attempted_ids)-4})"
                cand_label = f"{0 if final_data.empty else 1}/{len(cands) if len(cands)>0 else 0}"
                print(
                    f"[Pull firm {pull_idx}/{total}] firm_id={firm_id[:40]} | cand_used={cand_label} | "
                    f"coverage={cov_pct:.2f}% | index_range={index_range} | "
                    f"pulled_range={pulled_range} | tried_ids: {tried_preview}"
                )

                if final_data.empty:
                    missing_rows = {
                        "firm_id": firm_id,
                        "company_name": row.get("company_name", pd.NA),
                        "ISIN": row.get("ISIN", pd.NA),
                        "RIC": row.get("RIC", pd.NA),
                        "RIC_current": row.get("RIC_current", pd.NA),
                        "last_failed_at": pd.Timestamp.utcnow().isoformat(),
                        "reason": "no_returns_after_candidates",
                        "n_candidates": int(len(cands)),
                        "tried_ids": "|".join(attempted_ids),
                    }
                    bad_rows.append(missing_rows)
                    batch_bad_rows.append(missing_rows)
                else:
                    tmp = final_data.copy()
                    tmp["firm_id"] = firm_id
                    tmp["name"] = row.get("company_name", pd.NA)
                    tmp["pull_id"] = selected_pull_id
                    tmp["id_type"] = selected_id_type
                    all_company_returns.append(tmp)

                mrow = {
                    "firm_id": firm_id,
                    "pull_id": selected_pull_id,
                    "id_type": selected_id_type,
                    "n_rows": int(len(final_data)),
                    "date_min": final_data["date"].min() if not final_data.empty else pd.NaT,
                    "date_max": final_data["date"].max() if not final_data.empty else pd.NaT,
                    "coverage_rate_pct": cov_pct,
                    "field_used": field_used,
                    "mode_used": mode_used,
                    "cache_path": str(_dr_cache_path(cache_dir, firm_id)),
                    "status": "ok" if not final_data.empty else "failed",
                }
                manifest_rows.append(mrow)
                step_rows.append(mrow)
                batch_processed.append(str(firm_id))

                if (pull_idx % max(1, cfg.checkpoint_every_n_pulls)) == 0:
                    pd.DataFrame(step_rows).to_parquet(step_rows_path, index=False)
                    processed_companies.update(batch_processed)
                    step_ckpt_path.write_text(
                        json.dumps(
                            {
                                "processed_companies": sorted(processed_companies),
                                "remaining_companies": max(0, companies_total - len(processed_companies)),
                                "updated_at_utc": pd.Timestamp.utcnow().isoformat(),
                                "rows": int(len(step_rows)),
                            },
                            ensure_ascii=False,
                            indent=2,
                        )
                    )

            if batch_bad_rows:
                append_bad_ids_rows(bad_ids_path, batch_bad_rows)

            if batch_processed:
                processed_companies.update(batch_processed)
                step_ckpt_path.write_text(
                    json.dumps(
                        {
                            "processed_companies": sorted(processed_companies),
                            "remaining_companies": max(0, companies_total - len(processed_companies)),
                            "updated_at_utc": pd.Timestamp.utcnow().isoformat(),
                            "rows": int(len(step_rows)),
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                )

            if batch_rate_limit_abort and cfg.pull_abort_on_rate_limit:
                break
    finally:
        if not cfg.skip_lseg_pull:
            try:
                ld.close_session()
            except Exception:
                pass

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_parquet(manifest_path, index=False)

    if all_company_returns:
        returns_all = pd.concat(all_company_returns, ignore_index=True)
    else:
        returns_all = pd.DataFrame(columns=["date", "ret", "firm_id", "name", "pull_id", "id_type"])
    returns_all["date"] = pd.to_datetime(returns_all["date"], errors="coerce")
    returns_all = returns_all.dropna(subset=["date"]).copy()
    returns_all = returns_all[returns_all["date"] <= cfg.target_end_date].copy()
    returns_all = returns_all.sort_values(["firm_id", "date"]).reset_index(drop=True)
    returns_all.to_parquet(output_returns_all, index=False)

    missing_cols = [
        "firm_id",
        "company_name",
        "ISIN",
        "RIC",
        "RIC_current",
        "last_failed_at",
        "reason",
        "n_candidates",
        "tried_ids",
    ]
    if bad_rows:
        missing_df = pd.DataFrame(bad_rows)
        for c in missing_cols:
            if c not in missing_df.columns:
                missing_df[c] = pd.NA
        missing_df = missing_df[missing_cols].copy()
    else:
        missing_df = pd.DataFrame(columns=missing_cols)
    missing_df.to_parquet(output_missing, index=False)

    pd.DataFrame(step_rows).to_parquet(step_rows_path, index=False)
    step_ckpt_path.write_text(
        json.dumps(
            {
                "processed_companies": sorted(processed_companies),
                "remaining_companies": max(0, companies_total - len(processed_companies)),
                "updated_at_utc": pd.Timestamp.utcnow().isoformat(),
                "rows": int(len(step_rows)),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    bad_hist_after = append_bad_ids_rows(bad_ids_path, bad_rows)

    print("Saved manifest:", manifest_path)
    print("Saved company returns:", output_returns_all, "rows:", len(returns_all))
    print("Built missing list in-memory rows:", len(missing_df))
    print("Updated bad-id log:", bad_ids_path, "rows:", len(bad_hist_after))
    print("Saved step rows:", step_rows_path, "rows:", len(step_rows))
    print("Saved step checkpoint:", step_ckpt_path)
    post_cache_files = {f.name for f in cache_dir.glob("*.parquet")}
    print("New cache files created this run:", len(post_cache_files - initial_cache_files))
    print(
        f"Done: companies_total={companies_total}, run_remaining_start={total}, "
        f"resolved_rows={int(len(returns_all))}, unresolved_rows={int(len(missing_df))}, "
        f"full_coverage={full_coverage}, partial_coverage={partial_coverage}, bad_ids={bad_ids_count}"
    )

    return {
        "companies_total": int(companies_total),
        "full_coverage": int(full_coverage),
        "partial_coverage": int(partial_coverage),
        "bad_ids": int(bad_ids_count),
        "remaining": int(remaining),
        "manifest_rows": int(len(manifest_df)),
        "returns_rows": int(len(returns_all)),
        "missing_rows": int(len(missing_df)),
        "known_bad_ids": int(len(bad_hist_after)),
    }


def run_standard_pull(pull_type: str, **kwargs):
    """Unified entry-point for all standard pulls.

    pull_type:
    - 'series' -> internal series pipeline
    - 'daily_returns' -> internal daily-returns pipeline
    """
    key = str(pull_type).strip().lower()
    if key == "series":
        return _run_series_internal(**kwargs)
    if key == "daily_returns":
        return _run_daily_returns_internal(**kwargs)
    raise ValueError("pull_type must be one of: 'series', 'daily_returns'")
