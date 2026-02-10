from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _lazy_import_hf() -> tuple[Any, Any]:
    try:
        import torch  # type: ignore
        from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing optional deps for HF leader. Install with: pip install -e '.[hf]'"
        ) from e
    return torch, (AutoModelForSequenceClassification, AutoTokenizer)


def _resolve_device(torch: Any, device: str) -> Any:
    d = device.strip().lower()
    if d in {"cpu", "cuda", "mps"}:
        if d == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("device=cuda requested but CUDA is not available")
        if d == "mps" and not getattr(torch.backends, "mps", None):
            raise RuntimeError("device=mps requested but MPS backend is not available")
        if d == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("device=mps requested but MPS is not available")
        return torch.device(d)

    if d != "auto":
        raise ValueError("device must be one of: auto, cpu, mps, cuda")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _fallback_label_names(num_labels: int) -> list[str]:
    if num_labels == 6:
        return [
            "toxic",
            "severe_toxic",
            "obscene",
            "threat",
            "insult",
            "identity_hate",
        ]
    return [f"label_{i}" for i in range(num_labels)]


def _label_names_from_config(cfg: Any) -> list[str]:
    id2label = getattr(cfg, "id2label", None)
    if isinstance(id2label, dict) and id2label:
        keys = sorted(int(k) for k in id2label.keys())
        names = [str(id2label[i]) for i in keys]
        if all(n.startswith("LABEL_") for n in names):
            return _fallback_label_names(len(names))
        return [n.strip().lower() for n in names]
    return _fallback_label_names(int(getattr(cfg, "num_labels", 2)))


def _parse_weights(s: str | None, labels: list[str]) -> dict[str, float] | None:
    if s is None:
        return None
    txt = s.strip()
    if not txt:
        return None

    out: dict[str, float] = {}
    parts = [p.strip() for p in txt.split(",") if p.strip()]
    for p in parts:
        if "=" not in p:
            raise ValueError("hf_weights must look like: toxic=1.0,insult=0.5")
        k, v = p.split("=", 1)
        key = k.strip().lower()
        if key not in labels:
            raise ValueError(f"Unknown label in hf_weights: {key}. Labels: {labels}")
        out[key] = float(v.strip())
    if not out:
        return None
    return out


def _any_prob(p_list: list[float]) -> float:
    prod = 1.0
    for p in p_list:
        prod *= 1.0 - float(p)
    return float(1.0 - prod)


@dataclass(frozen=True)
class HfToxicityConfig:
    model_name: str = "Arsive/roberta-toxicity-classifier"
    device: str = "auto"
    batch_size: int = 8
    max_length: int = 256
    category_threshold: float = 0.5
    score_mode: str = "any"  # any|toxic|max|weighted
    weights: str | None = None  # e.g. "toxic=1.0,insult=0.5"
    weighted_normalize: bool = True
    weighted_clamp: bool = True


class HfToxicityLeader:
    def __init__(self, cfg: HfToxicityConfig) -> None:
        torch, (AutoModelForSequenceClassification, AutoTokenizer) = _lazy_import_hf()
        self._torch = torch
        self.cfg = cfg
        self._device = _resolve_device(torch, cfg.device)

        self._tok = AutoTokenizer.from_pretrained(cfg.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(cfg.model_name)
        self._model.to(self._device)
        self._model.eval()

        self._labels = _label_names_from_config(self._model.config)
        self._weights = _parse_weights(cfg.weights, self._labels)

        mode = cfg.score_mode.strip().lower()
        if mode not in {"any", "toxic", "max", "weighted"}:
            raise ValueError("score_mode must be one of: any,toxic,max,weighted")
        if mode == "weighted" and not self._weights:
            raise ValueError("score_mode=weighted requires --hf-weights")
        self._mode = mode

    def _score_scalar(self, label_probs: dict[str, float]) -> float:
        probs = list(label_probs.values())
        if not probs:
            return 0.0

        if self._mode == "max":
            return float(max(probs))

        if self._mode == "any":
            return _any_prob(probs)

        if self._mode == "toxic":
            if "toxic" in label_probs:
                return float(label_probs["toxic"])
            return float(max(probs))

        # weighted
        assert self._weights is not None
        raw = 0.0
        wsum = 0.0
        for k, w in self._weights.items():
            ww = float(w)
            if ww < 0.0:
                raise ValueError("hf_weights must be non-negative to guarantee a [0,1] score")
            wsum += ww
            raw += ww * float(label_probs.get(k, 0.0))

        if wsum <= 0.0:
            score = 0.0
        elif self.cfg.weighted_normalize:
            score = raw / wsum
        else:
            score = raw

        if self.cfg.weighted_clamp:
            if score < 0.0:
                score = 0.0
            if score > 1.0:
                score = 1.0

        return float(score)

    def score(self, x: str, y: str | None = None):
        text = y if y is not None else x
        return self.score_many([text])[0]

    def score_many(self, texts: list[str]):
        from bias_stackelberg.core.types import BiasReport  # noqa: PLC0415

        torch = self._torch
        bs = max(1, int(self.cfg.batch_size))
        out: list[BiasReport] = []

        with torch.no_grad():
            for i in range(0, len(texts), bs):
                chunk = texts[i : i + bs]
                enc = self._tok(
                    chunk,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=int(self.cfg.max_length),
                )
                enc = {k: v.to(self._device) for k, v in enc.items()}
                logits = self._model(**enc).logits

                probs_t = torch.sigmoid(logits).detach().float().cpu().tolist()

                for p in probs_t:
                    p_list = [float(x) for x in p]
                    pairs = list(zip(self._labels, p_list, strict=False))
                    label_probs = {name: float(val) for name, val in pairs}

                    score = self._score_scalar(label_probs)
                    cats = [
                        name
                        for name, val in pairs
                        if float(val) >= float(self.cfg.category_threshold)
                    ]

                    trace = {
                        "leader": "HfToxicityLeader",
                        "model_name": self.cfg.model_name,
                        "device": str(self._device),
                        "category_threshold": float(self.cfg.category_threshold),
                        "score_mode": self._mode,
                        "weights": self._weights,
                        "label_probs": label_probs,
                    }

                    out.append(
                        BiasReport(
                            score=float(score),
                            categories=cats,
                            evidence_spans=[],
                            rationale="HF multi-label toxicity score",
                            trace=trace,
                        )
                    )

        return out
