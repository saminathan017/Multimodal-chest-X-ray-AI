"""
Persistent clinical worklist store.

This module stores de-identified case metadata, AI outputs, review state, and
clinician feedback in SQLite. It is deliberately small but production-shaped:
the interface can later be backed by Postgres, FHIR resources, or a hospital
worklist service without changing API route behavior.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class CaseRecord:
    case_id: str
    request_id: str
    created_at: str
    updated_at: str
    status: str
    priority: str
    urgency_score: float
    top_finding: str
    assigned_to: str | None
    patient_context: dict[str, Any]
    findings: list[dict[str, Any]]
    workflow: dict[str, Any]
    uncertainty: dict[str, Any]
    clinical_report: str
    safety_flags: list[str]
    integration: dict[str, Any]
    structured_findings: list[dict[str, Any]]
    report_versions: list[dict[str, Any]]
    feedback: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "request_id": self.request_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "status": self.status,
            "priority": self.priority,
            "urgency_score": self.urgency_score,
            "top_finding": self.top_finding,
            "assigned_to": self.assigned_to,
            "patient_context": self.patient_context,
            "findings": self.findings,
            "workflow": self.workflow,
            "uncertainty": self.uncertainty,
            "clinical_report": self.clinical_report,
            "safety_flags": self.safety_flags,
            "integration": self.integration,
            "structured_findings": self.structured_findings,
            "report_versions": self.report_versions,
            "feedback": self.feedback,
        }


class CaseStore:
    VALID_STATUSES = {"new", "in_review", "accepted", "edited", "rejected", "escalated"}

    def __init__(self, db_path: str | Path = "data/clinical_worklist.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cases (
                    case_id TEXT PRIMARY KEY,
                    request_id TEXT UNIQUE NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    urgency_score REAL NOT NULL,
                    top_finding TEXT NOT NULL,
                    assigned_to TEXT,
                    patient_context_json TEXT NOT NULL,
                    findings_json TEXT NOT NULL,
                    workflow_json TEXT NOT NULL,
                    uncertainty_json TEXT NOT NULL,
                    clinical_report TEXT NOT NULL,
                    safety_flags_json TEXT NOT NULL,
                    integration_json TEXT NOT NULL DEFAULT '{}',
                    structured_findings_json TEXT NOT NULL DEFAULT '[]'
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback (
                    feedback_id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    user_id_hash TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    corrected INTEGER NOT NULL,
                    radiologist_findings_json TEXT NOT NULL,
                    comments TEXT,
                    FOREIGN KEY(case_id) REFERENCES cases(case_id)
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cases_status ON cases(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cases_priority ON cases(priority)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cases_created ON cases(created_at)")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS report_versions (
                    version_id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    author_id_hash TEXT NOT NULL,
                    source TEXT NOT NULL,
                    report_text TEXT NOT NULL,
                    structured_findings_json TEXT NOT NULL,
                    change_summary TEXT,
                    FOREIGN KEY(case_id) REFERENCES cases(case_id)
                )
                """
            )
            self._ensure_column(conn, "cases", "integration_json", "TEXT NOT NULL DEFAULT '{}'")
            self._ensure_column(conn, "cases", "structured_findings_json", "TEXT NOT NULL DEFAULT '[]'")

    def _ensure_column(
        self,
        conn: sqlite3.Connection,
        table: str,
        column: str,
        definition: str,
    ) -> None:
        columns = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        if column not in columns:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def create_case(
        self,
        *,
        request_id: str,
        priority: str,
        urgency_score: float,
        top_finding: str,
        patient_context: dict[str, Any] | None,
        findings: list[dict[str, Any]],
        workflow: dict[str, Any],
        uncertainty: dict[str, Any],
        clinical_report: str,
        safety_flags: list[str],
        integration: dict[str, Any] | None = None,
        structured_findings: list[dict[str, Any]] | None = None,
    ) -> CaseRecord:
        now = utc_now()
        case_id = f"case-{uuid.uuid4().hex[:12]}"
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO cases (
                    case_id, request_id, created_at, updated_at, status, priority,
                    urgency_score, top_finding, assigned_to, patient_context_json,
                    findings_json, workflow_json, uncertainty_json, clinical_report,
                    safety_flags_json, integration_json, structured_findings_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    case_id,
                    request_id,
                    now,
                    now,
                    "new",
                    priority,
                    urgency_score,
                    top_finding,
                    None,
                    json.dumps(patient_context or {}),
                    json.dumps(findings),
                    json.dumps(workflow),
                    json.dumps(uncertainty),
                    clinical_report,
                    json.dumps(safety_flags),
                    json.dumps(integration or {}),
                    json.dumps(structured_findings or self._default_structured_findings(findings)),
                ),
            )
        created = self.get_case(case_id)
        self.add_report_version(
            case_id=created.case_id,
            author_id_hash="ai-system",
            source="ai_draft",
            report_text=clinical_report,
            structured_findings=created.structured_findings,
            change_summary="Initial AI draft",
        )
        return self.get_case(case_id)

    def list_cases(
        self,
        *,
        status: str | None = None,
        priority: str | None = None,
        limit: int = 50,
    ) -> list[CaseRecord]:
        clauses: list[str] = []
        params: list[Any] = []
        if status:
            clauses.append("status = ?")
            params.append(status)
        if priority:
            clauses.append("priority = ?")
            params.append(priority)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM cases {where} ORDER BY urgency_score DESC, created_at DESC LIMIT ?",
                params,
            ).fetchall()
        return [self._row_to_case(row) for row in rows]

    def get_case(self, case_id: str) -> CaseRecord:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,)).fetchone()
        if row is None:
            raise KeyError(case_id)
        return self._row_to_case(row)

    def get_case_by_request_id(self, request_id: str) -> CaseRecord:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM cases WHERE request_id = ?", (request_id,)).fetchone()
        if row is None:
            raise KeyError(request_id)
        return self._row_to_case(row)

    def update_status(
        self,
        case_id: str,
        *,
        status: str,
        assigned_to: str | None = None,
    ) -> CaseRecord:
        if status not in self.VALID_STATUSES:
            raise ValueError(f"Invalid status: {status}")
        with self._connect() as conn:
            cur = conn.execute(
                """
                UPDATE cases
                SET status = ?, assigned_to = COALESCE(?, assigned_to), updated_at = ?
                WHERE case_id = ?
                """,
                (status, assigned_to, utc_now(), case_id),
            )
            if cur.rowcount == 0:
                raise KeyError(case_id)
        return self.get_case(case_id)

    def update_integration(
        self,
        case_id: str,
        *,
        integration: dict[str, Any],
    ) -> CaseRecord:
        with self._connect() as conn:
            cur = conn.execute(
                """
                UPDATE cases
                SET integration_json = ?, updated_at = ?
                WHERE case_id = ?
                """,
                (json.dumps(integration), utc_now(), case_id),
            )
            if cur.rowcount == 0:
                raise KeyError(case_id)
        return self.get_case(case_id)

    def add_report_version(
        self,
        *,
        case_id: str,
        author_id_hash: str,
        source: str,
        report_text: str,
        structured_findings: list[dict[str, Any]],
        change_summary: str | None = None,
    ) -> dict[str, Any]:
        self.get_case(case_id)
        version_id = f"rv-{uuid.uuid4().hex[:12]}"
        now = utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO report_versions (
                    version_id, case_id, created_at, author_id_hash, source,
                    report_text, structured_findings_json, change_summary
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    version_id,
                    case_id,
                    now,
                    author_id_hash,
                    source,
                    report_text,
                    json.dumps(structured_findings),
                    change_summary,
                ),
            )
            conn.execute(
                """
                UPDATE cases
                SET clinical_report = ?, structured_findings_json = ?, updated_at = ?, status = 'edited'
                WHERE case_id = ?
                """,
                (report_text, json.dumps(structured_findings), now, case_id),
            )
        return {
            "version_id": version_id,
            "case_id": case_id,
            "created_at": now,
            "author_id_hash": author_id_hash,
            "source": source,
            "report_text": report_text,
            "structured_findings": structured_findings,
            "change_summary": change_summary,
        }

    def list_report_versions(self, case_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM report_versions WHERE case_id = ? ORDER BY created_at DESC",
                (case_id,),
            ).fetchall()
        return [self._version_row_to_dict(row) for row in rows]

    def add_feedback(
        self,
        *,
        request_id: str,
        user_id_hash: str,
        decision: str,
        corrected: bool,
        radiologist_findings: list[str],
        comments: str | None,
    ) -> dict[str, Any]:
        case = self.get_case_by_request_id(request_id)
        feedback_id = f"fb-{uuid.uuid4().hex[:12]}"
        entry = {
            "feedback_id": feedback_id,
            "case_id": case.case_id,
            "request_id": request_id,
            "created_at": utc_now(),
            "user_id_hash": user_id_hash,
            "decision": decision,
            "corrected": corrected,
            "radiologist_findings": radiologist_findings,
            "comments": comments,
        }
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO feedback (
                    feedback_id, case_id, request_id, created_at, user_id_hash,
                    decision, corrected, radiologist_findings_json, comments
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    feedback_id,
                    case.case_id,
                    request_id,
                    entry["created_at"],
                    user_id_hash,
                    decision,
                    int(corrected),
                    json.dumps(radiologist_findings),
                    comments,
                ),
            )
        return entry

    def _row_to_case(self, row: sqlite3.Row) -> CaseRecord:
        with self._connect() as conn:
            feedback_rows = conn.execute(
                "SELECT * FROM feedback WHERE case_id = ? ORDER BY created_at DESC",
                (row["case_id"],),
            ).fetchall()
        feedback = [
            {
                "feedback_id": item["feedback_id"],
                "case_id": item["case_id"],
                "request_id": item["request_id"],
                "created_at": item["created_at"],
                "user_id_hash": item["user_id_hash"],
                "decision": item["decision"],
                "corrected": bool(item["corrected"]),
                "radiologist_findings": json.loads(item["radiologist_findings_json"]),
                "comments": item["comments"],
            }
            for item in feedback_rows
        ]
        return CaseRecord(
            case_id=row["case_id"],
            request_id=row["request_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            status=row["status"],
            priority=row["priority"],
            urgency_score=float(row["urgency_score"]),
            top_finding=row["top_finding"],
            assigned_to=row["assigned_to"],
            patient_context=json.loads(row["patient_context_json"]),
            findings=json.loads(row["findings_json"]),
            workflow=json.loads(row["workflow_json"]),
            uncertainty=json.loads(row["uncertainty_json"]),
            clinical_report=row["clinical_report"],
            safety_flags=json.loads(row["safety_flags_json"]),
            integration=json.loads(row["integration_json"]),
            structured_findings=json.loads(row["structured_findings_json"]),
            report_versions=self.list_report_versions(row["case_id"]),
            feedback=feedback,
        )

    def _version_row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "version_id": row["version_id"],
            "case_id": row["case_id"],
            "created_at": row["created_at"],
            "author_id_hash": row["author_id_hash"],
            "source": row["source"],
            "report_text": row["report_text"],
            "structured_findings": json.loads(row["structured_findings_json"]),
            "change_summary": row["change_summary"],
        }

    def _default_structured_findings(self, findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
        structured = []
        for item in findings:
            prob = float(item.get("prob", 0.0))
            structured.append(
                {
                    "label": item.get("label", "Finding"),
                    "status": "suspected" if prob >= 0.45 else "not_confirmed",
                    "probability": round(prob, 4),
                    "laterality": "unspecified",
                    "location": "unspecified",
                    "severity": "high" if prob >= 0.75 else ("moderate" if prob >= 0.45 else "low"),
                    "clinician_note": "",
                }
            )
        return structured
