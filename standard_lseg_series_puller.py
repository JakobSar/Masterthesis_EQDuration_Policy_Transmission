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
    batch_size: int = 20
    batch_pause_sec: float = 0.0
    min_asof_date: pd.Timestamp = pd.Timestamp("1900-01-01")
    force_refresh: bool = False
    cache_only: bool = False


def _clean_str(s: pd.Series) -> pd.Series:
    x = s.astype("string").str.strip()
    return x.where(x.notna() & (x != ""), pd.NA)


def _safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(text))


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_bad_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        d = pd.read_csv(path)
        if "bad_id_key" not in d.columns:
            return set()
        return set(d["bad_id_key"].dropna().astype(str).str.strip().tolist())
    except Exception:
        return set()


def save_bad_ids(path: Path, bad_ids: set[str]) -> None:
    _ensure_parent(path)
    out = pd.DataFrame({"bad_id_key": sorted(set(str(x).strip() for x in bad_ids if str(x).strip()))})
    tmp = path.with_suffix(path.suffix + ".tmp")
    out.to_csv(tmp, index=False)
    tmp.replace(path)


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


def build_company_candidates(company_req: pd.DataFrame) -> list[tuple[str, str]]:
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

    for col, id_type in [("ISIN", "ISIN"), ("RIC_current", "RIC"), ("RIC", "RIC")]:
        if col in q.columns:
            for v in q[col].dropna().astype(str):
                _add(id_type, v)

    if {"id_type", "pull_id"}.issubset(q.columns):
        for _, row in q[["id_type", "pull_id"]].dropna().iterrows():
            it = str(row["id_type"]).strip().upper()
            pid = str(row["pull_id"]).strip()
            if not pid:
                continue
            if it == "ISIN":
                _add("ISIN", pid)
                _add("ISIN", f"ISIN:{pid}")
            elif it == "RIC":
                _add("RIC", pid)

    return out


def extract_history(raw: pd.DataFrame, output_col: str, field_hint: str) -> pd.DataFrame:
    if raw is None or len(raw) == 0:
        return pd.DataFrame(columns=["date", output_col])

    x = raw.copy()
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
    val_col = None

    for c in value_cols:
        uc = str(c).upper().replace(" ", "")
        if field_hint.upper().replace(" ", "") in uc:
            val_col = c
            break

    if val_col is None and len(value_cols) == 1:
        val_col = value_cols[0]

    if val_col is None:
        for c in value_cols:
            sc = pd.to_numeric(x[c], errors="coerce")
            if int(sc.notna().sum()) > 0:
                val_col = c
                break

    if val_col is None:
        return pd.DataFrame(columns=["date", output_col])

    out = pd.DataFrame(
        {
            "date": pd.to_datetime(x[date_col], errors="coerce").dt.normalize(),
            output_col: pd.to_numeric(x[val_col], errors="coerce"),
        }
    )
    out = out.dropna(subset=["date"]).sort_values("date").drop_duplicates(["date"], keep="last")
    return out[["date", output_col]]


def map_history_to_asof(req_dates: pd.Series, hist: pd.DataFrame, output_col: str, tol_days: int) -> pd.DataFrame:
    left = pd.DataFrame({"date": pd.to_datetime(req_dates, errors="coerce").dt.normalize()}).dropna().sort_values("date")
    if left.empty:
        return pd.DataFrame(columns=["date", output_col])
    if hist is None or hist.empty:
        left[output_col] = np.nan
        return left

    right = hist[["date", output_col]].copy()
    right["date"] = pd.to_datetime(right["date"], errors="coerce").dt.normalize()
    right[output_col] = pd.to_numeric(right[output_col], errors="coerce")
    right = right.dropna(subset=["date"]).sort_values("date").drop_duplicates(["date"], keep="last")

    return pd.merge_asof(
        left,
        right,
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
        bad_rows_log_path: Path,
    ) -> None:
        self.cfg = config
        self.cache_dir = cache_dir
        self.step_rows_path = step_rows_path
        self.checkpoint_path = checkpoint_path
        self.bad_ids_path = bad_ids_path
        self.bad_rows_log_path = bad_rows_log_path
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        _ensure_parent(self.step_rows_path)
        _ensure_parent(self.checkpoint_path)
        _ensure_parent(self.bad_ids_path)
        _ensure_parent(self.bad_rows_log_path)

    def _cache_path_for_company_id(self, firm_id: str, id_type: str, pull_id: str) -> Path:
        base = _safe_name(firm_id).replace(".parquet", "")
        suffix = _safe_name(f"{id_type}_{pull_id}")
        return self.cache_dir / f"{base}__{suffix}.parquet"

    def _load_cache(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame(columns=["date", self.cfg.output_col])
        try:
            d = pd.read_parquet(path).copy()
        except Exception:
            return pd.DataFrame(columns=["date", self.cfg.output_col])
        if "date" not in d.columns:
            return pd.DataFrame(columns=["date", self.cfg.output_col])
        if self.cfg.output_col not in d.columns:
            d[self.cfg.output_col] = np.nan
        d["date"] = pd.to_datetime(d["date"], errors="coerce").dt.normalize()
        d[self.cfg.output_col] = pd.to_numeric(d[self.cfg.output_col], errors="coerce")
        d = d.dropna(subset=["date"]).sort_values("date").drop_duplicates(["date"], keep="last")
        return d[["date", self.cfg.output_col]]

    def _save_cache(self, path: Path, df: pd.DataFrame) -> None:
        d = df.copy()
        d["date"] = pd.to_datetime(d["date"], errors="coerce").dt.normalize()
        d[self.cfg.output_col] = pd.to_numeric(d[self.cfg.output_col], errors="coerce")
        d = d.dropna(subset=["date"]).sort_values("date").drop_duplicates(["date"], keep="last")
        _ensure_parent(path)
        tmp = path.with_suffix(path.suffix + ".tmp")
        d.to_parquet(tmp, index=False)
        tmp.replace(path)

    def _pull_segment(self, pull_id: str, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, bool]:
        if pd.isna(start) or pd.isna(end) or start > end:
            return pd.DataFrame(columns=["date", self.cfg.output_col]), False

        fields = [self.cfg.field]
        if self.cfg.fallback_field and self.cfg.fallback_field != self.cfg.field:
            fields.append(self.cfg.fallback_field)

        for field_name in fields:
            for interval in self.cfg.intervals:
                last_err = None
                for r in range(self.cfg.max_retries):
                    try:
                        raw = ld.get_history(
                            universe=[pull_id],
                            fields=[field_name],
                            start=start.strftime("%Y-%m-%d"),
                            end=end.strftime("%Y-%m-%d"),
                            interval=interval,
                        )
                        out = extract_history(raw, output_col=self.cfg.output_col, field_hint=self.cfg.field)
                        if not out.empty:
                            return out, False
                        break
                    except Exception as e:
                        last_err = e
                        msg = str(e)
                        if "Unable to resolve all requested identifiers" in msg:
                            return pd.DataFrame(columns=["date", self.cfg.output_col]), True
                        time.sleep(self.cfg.base_sleep_sec * (2**r) + random.random() * 0.3)
                if last_err is not None:
                    continue
        return pd.DataFrame(columns=["date", self.cfg.output_col]), False

    def _update_company_cache(
        self, firm_id: str, id_type: str, pull_id: str, start: pd.Timestamp, end: pd.Timestamp
    ) -> tuple[pd.DataFrame, bool]:
        path = self._cache_path_for_company_id(firm_id, id_type, pull_id)
        cached = pd.DataFrame(columns=["date", self.cfg.output_col]) if self.cfg.force_refresh else self._load_cache(path)

        has_values = bool(pd.to_numeric(cached.get(self.cfg.output_col), errors="coerce").notna().any()) if not cached.empty else False
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

        pulled, permanent_bad_id = self._pull_segment(pull_id=pull_id, start=start, end=end)
        frames = [x for x in [cached, pulled] if not x.empty]
        out = (
            pd.concat(frames, ignore_index=True).sort_values("date").drop_duplicates(["date"], keep="last")
            if frames
            else pd.DataFrame(columns=["date", self.cfg.output_col])
        )
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
            str(fid): build_company_candidates(g) for fid, g in req.groupby("firm_id", sort=False)
        }
        companies_all = req["firm_id"].dropna().astype(str).unique().tolist()
        companies_total = len(companies_all)

        existing_step_rows = (
            normalize_step_rows(pd.read_parquet(self.step_rows_path), self.cfg.output_col)
            if self.step_rows_path.exists()
            else pd.DataFrame(columns=["firm_id", "date", self.cfg.output_col, "rank", "id_type", "pull_id"])
        )
        processed_from_rows = set(
            existing_step_rows.loc[pd.to_numeric(existing_step_rows[self.cfg.output_col], errors="coerce").notna(), "firm_id"]
            .dropna()
            .astype(str)
            .tolist()
        )

        processed_from_ckpt = set()
        if self.checkpoint_path.exists():
            try:
                ck = json.loads(self.checkpoint_path.read_text())
                processed_from_ckpt = set(str(x) for x in ck.get("processed_companies", []) if str(x).strip())
            except Exception:
                processed_from_ckpt = set()
        if processed_from_ckpt and processed_from_rows:
            processed_from_ckpt = processed_from_ckpt & processed_from_rows
        else:
            processed_from_ckpt = set()

        processed_companies = set(processed_from_rows) | set(processed_from_ckpt)
        companies = [c for c in companies_all if str(c) not in processed_companies]
        bad_ids = load_bad_ids(self.bad_ids_path)

        print("\n" + "=" * 88)
        print("Standard Series Pull Overview")
        print("=" * 88)
        print(f"field: {self.cfg.field} -> output_col: {self.cfg.output_col}")
        print(f"request_rows: {len(req):,} | companies_total: {companies_total:,} | remaining: {len(companies):,}")
        print(f"mode: {'CACHE_ONLY' if self.cfg.cache_only else 'CACHE+NETWORK'} | batch_size: {self.cfg.batch_size}")
        print(f"known_bad_ids: {len(bad_ids):,} ({self.bad_ids_path.name})")
        print("=" * 88)

        total_cand_calls = 0
        total_resolved = 0
        total_unresolved = 0
        total_bad_id_skips = 0
        total_bad_ids_added = 0
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
                print(f"[BATCH {b_ix}/{n_batches}] companies={len(batch_companies)} idx={b_start+1}-{b_end}")

                for k, firm_id in enumerate(batch_companies, start=1):
                    company_req = req[req["firm_id"] == firm_id].copy().sort_values("date")
                    req_dates = company_req["date"].dropna().drop_duplicates().sort_values().reset_index(drop=True)
                    if req_dates.empty:
                        continue
                    start = pd.to_datetime(req_dates.min()).normalize()
                    end = pd.to_datetime(req_dates.max()).normalize()

                    panel = pd.DataFrame({"date": req_dates})
                    panel[self.cfg.output_col] = np.nan
                    panel["rank"] = pd.NA
                    panel["id_type"] = pd.NA
                    panel["pull_id"] = pd.NA

                    attempted_ids: list[str] = []
                    cands = company_candidates_map.get(str(firm_id), [])
                    cand_used = 0
                    skipped_bad = 0

                    for rank, (cand_type, cand_id) in enumerate(cands, start=1):
                        if panel[self.cfg.output_col].notna().all():
                            break
                        cand_type = str(cand_type).upper().strip()
                        cand_id = str(cand_id).strip()
                        if not cand_type or not cand_id:
                            continue

                        bad_key = f"{cand_type}:{cand_id}"
                        if bad_key in bad_ids:
                            skipped_bad += 1
                            continue

                        cand_used += 1
                        total_cand_calls += 1
                        attempted_ids.append(bad_key)

                        hist, permanent_bad_id = self._update_company_cache(
                            firm_id=str(firm_id),
                            id_type=cand_type,
                            pull_id=cand_id,
                            start=start,
                            end=end,
                        )

                        if permanent_bad_id:
                            if bad_key not in bad_ids:
                                bad_ids.add(bad_key)
                                total_bad_ids_added += 1
                            continue

                        mapped = map_history_to_asof(req_dates, hist, output_col=self.cfg.output_col, tol_days=self.cfg.asof_tolerance_days)
                        panel = panel.merge(mapped.rename(columns={self.cfg.output_col: f"{self.cfg.output_col}_cand"}), on="date", how="left")
                        fill_mask = panel[self.cfg.output_col].isna() & panel[f"{self.cfg.output_col}_cand"].notna()
                        panel.loc[fill_mask, self.cfg.output_col] = panel.loc[fill_mask, f"{self.cfg.output_col}_cand"]
                        panel.loc[fill_mask, "rank"] = rank
                        panel.loc[fill_mask, "id_type"] = cand_type
                        panel.loc[fill_mask, "pull_id"] = cand_id
                        panel = panel.drop(columns=[f"{self.cfg.output_col}_cand"])

                    total_bad_id_skips += skipped_bad
                    panel["firm_id"] = str(firm_id)
                    panel = panel[["firm_id", "date", self.cfg.output_col, "rank", "id_type", "pull_id"]]

                    resolved = int(panel[self.cfg.output_col].notna().sum())
                    unresolved = int(panel[self.cfg.output_col].isna().sum())
                    total_resolved += resolved
                    total_unresolved += unresolved

                    if unresolved > 0:
                        miss = panel[panel[self.cfg.output_col].isna()][["firm_id", "date"]].copy()
                        miss["reason"] = "no_data_after_fallback"
                        miss["n_candidates"] = len(cands)
                        bad_rows.extend(miss.to_dict("records"))

                    batch_new_rows.extend(panel.to_dict("records"))
                    batch_processed.append(str(firm_id))

                    resolved_dates = pd.to_datetime(panel.loc[panel[self.cfg.output_col].notna(), "date"], errors="coerce").dropna()
                    pulled_range = "NA:NA" if resolved_dates.empty else f"{resolved_dates.min().date()}:{resolved_dates.max().date()}"
                    range_in_index = f"{start.date()}:{end.date()}"
                    tried_preview = " | ".join(dict.fromkeys(attempted_ids)) if attempted_ids else "NA"
                    print(
                        f"[BATCH {b_ix}/{n_batches}] [{b_start+k}/{len(companies)}] "
                        f"firm_id={firm_id} | cand_used={cand_used}/{len(cands)} | bad_id_skip={skipped_bad} | "
                        f"unresolved={unresolved} | found_{self.cfg.output_col}={resolved} | "
                        f"range_in_index={range_in_index} | pulled_range={pulled_range} | tried_ids: {tried_preview}"
                    )

                if batch_new_rows:
                    batch_df = normalize_step_rows(pd.DataFrame(batch_new_rows), self.cfg.output_col)
                    prev = (
                        normalize_step_rows(pd.read_parquet(self.step_rows_path), self.cfg.output_col)
                        if self.step_rows_path.exists()
                        else pd.DataFrame(columns=batch_df.columns)
                    )
                    combined = (
                        pd.concat([prev, batch_df], ignore_index=True)
                        .sort_values(["firm_id", "date"])
                        .drop_duplicates(["firm_id", "date"], keep="last")
                    )
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
                    save_bad_ids(self.bad_ids_path, bad_ids)

                if self.cfg.batch_pause_sec > 0 and b_ix < n_batches:
                    time.sleep(self.cfg.batch_pause_sec)
        finally:
            if not self.cfg.cache_only:
                try:
                    ld.close_session()
                except Exception:
                    pass

        if bad_rows:
            bad_df = pd.DataFrame(bad_rows)
            if self.bad_rows_log_path.exists():
                old = pd.read_csv(self.bad_rows_log_path)
                out = pd.concat([old, bad_df], ignore_index=True)
            else:
                out = bad_df
            out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
            out = out.drop_duplicates(subset=["firm_id", "date", "reason"], keep="last")
            out.to_csv(self.bad_rows_log_path, index=False)

        print(
            f"Done: companies_total={companies_total}, run_remaining_start={len(companies)}, candidate_calls={total_cand_calls}, "
            f"resolved_rows={total_resolved}, unresolved_rows={total_unresolved}, found_{self.cfg.output_col}={total_resolved}, "
            f"bad_id_skip={total_bad_id_skips}, bad_ids_added={total_bad_ids_added}, known_bad_ids_now={len(bad_ids)}"
        )

        return {
            "companies_total": companies_total,
            "run_remaining_start": len(companies),
            "candidate_calls": total_cand_calls,
            "resolved_rows": total_resolved,
            "unresolved_rows": total_unresolved,
            "bad_id_skip": total_bad_id_skips,
            "bad_ids_added": total_bad_ids_added,
            "known_bad_ids_now": len(bad_ids),
        }


def build_ltg_request_rows(source_df: pd.DataFrame, output_col: str = "LTG") -> pd.DataFrame:
    req_cols = [c for c in ["firm_id", "date", "ISIN", "RIC_current", "RIC", "id_type", "pull_id", output_col] if c in source_df.columns]
    if "firm_id" not in req_cols or "date" not in req_cols:
        raise ValueError("source_df must contain at least firm_id and date.")
    req = source_df[req_cols].copy()
    req["date"] = pd.to_datetime(req["date"], errors="coerce").dt.normalize()
    req = req.dropna(subset=["firm_id", "date"]).drop_duplicates(["firm_id", "date"], keep="last")
    return req.reset_index(drop=True)


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
    skip_known_bad_ids: bool = True
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


def run_daily_returns_standard_puller(
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

    initial_cache_files = {f.name for f in cache_dir.glob("*.parquet")}
    bad_known = load_bad_ids(bad_ids_path) if (cfg.skip_known_bad_ids and not cfg.force_refresh) else set()
    step_rows: list[dict] = []
    manifest_rows: list[dict] = []
    bad_rows: list[dict] = []
    all_company_returns: list[pd.DataFrame] = []

    if not cfg.skip_lseg_pull:
        ld.open_session()
    try:
        pull_rows = cpm.copy()
        total = len(pull_rows)
        print(f"- total: {total} | cache_files: {len(initial_cache_files)}")
        print(f"- known_bad_ids: {len(bad_known)}")

        for pull_idx, (_, row) in enumerate(pull_rows.iterrows(), start=1):
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
            bad_skip = 0

            for cand_type, cand_id in cands:
                cand_type = str(cand_type).upper().strip()
                cand_id = str(cand_id).strip()
                if not cand_type or not cand_id:
                    continue
                bad_key = f"{cand_type}:{cand_id}"
                if bad_key in bad_known:
                    bad_skip += 1
                    continue
                attempted_ids.append(bad_key)
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
                    if _dr_is_unable_to_resolve_error(e):
                        bad_known.add(bad_key)
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
                f"bad_id_skip={bad_skip} | coverage={cov_pct:.2f}% | index_range={index_range} | "
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
                "bad_id_skip": int(bad_skip),
            }
            manifest_rows.append(mrow)
            step_rows.append(mrow)

            if (pull_idx % max(1, cfg.checkpoint_every_n_pulls)) == 0:
                pd.DataFrame(step_rows).to_parquet(step_rows_path, index=False)
                step_ckpt_path.write_text(
                    json.dumps({"processed_count": pull_idx, "updated_at_utc": pd.Timestamp.utcnow().isoformat()}, ensure_ascii=False, indent=2)
                )
                save_bad_ids(bad_ids_path, bad_known)
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

    missing_df = pd.DataFrame(bad_rows)
    missing_df.to_parquet(output_missing, index=False)

    pd.DataFrame(step_rows).to_parquet(step_rows_path, index=False)
    step_ckpt_path.write_text(
        json.dumps({"processed_count": len(step_rows), "updated_at_utc": pd.Timestamp.utcnow().isoformat()}, ensure_ascii=False, indent=2)
    )
    save_bad_ids(bad_ids_path, bad_known)

    print("Saved manifest:", manifest_path)
    print("Saved company returns:", output_returns_all, "rows:", len(returns_all))
    print("Built missing list in-memory rows:", len(missing_df))
    print("Updated bad-id log:", bad_ids_path, "rows:", len(bad_known))
    print("Saved step rows:", step_rows_path, "rows:", len(step_rows))
    print("Saved step checkpoint:", step_ckpt_path)
    post_cache_files = {f.name for f in cache_dir.glob("*.parquet")}
    print("New cache files created this run:", len(post_cache_files - initial_cache_files))

    return {
        "manifest_rows": int(len(manifest_df)),
        "returns_rows": int(len(returns_all)),
        "missing_rows": int(len(missing_df)),
        "known_bad_ids": int(len(bad_known)),
    }
