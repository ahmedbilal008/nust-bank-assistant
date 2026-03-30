import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent


class MLflowTracker:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.available = False
        self.mlflow = None

        if not enabled:
            return

        try:
            import mlflow  # type: ignore

            default_uri = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
            tracking_uri = os.getenv("MLFLOW_TRACKING_URI", default_uri)
            mlflow.set_tracking_uri(tracking_uri)
            self.mlflow = mlflow
            self.available = True
        except Exception:
            self.available = False

    def _set_experiment(self, name: str) -> None:
        if not self.available:
            return
        self.mlflow.set_experiment(name)

    def log_data_governance(self, version_entry: dict[str, Any], profile_path: Path) -> None:
        if not self.available:
            return

        self._set_experiment("nust-bank-data-governance")
        with self.mlflow.start_run(run_name=f"data_{version_entry['version']}"):
            self.mlflow.log_param("version", version_entry["version"])
            self.mlflow.log_param("tag", version_entry.get("tag", ""))
            self.mlflow.log_param("pairs", version_entry["records"].get("pairs", 0))
            self.mlflow.log_param("chunks", version_entry["records"].get("chunks", 0))

            for key, value in version_entry.get("hashes", {}).items():
                self.mlflow.log_param(key, value if value is not None else "")

            self.mlflow.log_artifact(str(profile_path), artifact_path="profiles")

    def log_inference(
        self,
        *,
        query: str,
        in_domain: bool,
        top_dense_score: float,
        retrieved_count: int,
        reranked_count: int,
        latency_ms: int,
        blocked_reason: str,
        used_adapter: bool,
        base_model: str,
    ) -> None:
        if not self.available:
            return

        self._set_experiment("nust-bank-inference")
        with self.mlflow.start_run(run_name="inference", nested=True):
            self.mlflow.log_param("base_model", base_model)
            self.mlflow.log_param("used_adapter", used_adapter)
            self.mlflow.log_param("query_len", len(query.split()))
            self.mlflow.log_param("in_domain", in_domain)
            self.mlflow.log_param("blocked_reason", blocked_reason)
            self.mlflow.log_metric("top_dense_score", float(top_dense_score))
            self.mlflow.log_metric("retrieved_count", int(retrieved_count))
            self.mlflow.log_metric("reranked_count", int(reranked_count))
            self.mlflow.log_metric("latency_ms", int(latency_ms))
