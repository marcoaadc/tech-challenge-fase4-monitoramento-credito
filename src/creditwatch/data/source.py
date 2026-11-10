"""Obtencao do dataset bruto com verificacao de integridade.

O snapshot em CSV fica versionado em ``data/raw/`` (licenca CC BY 4.0) para que o pipeline rode
offline. Se o arquivo nao existir, ele e baixado da UCI e convertido. O SHA-256 e conferido nos
dois pontos (o zip baixado e o CSV lido), de modo que qualquer alteracao na fonte ou no snapshot
interrompe o pipeline em vez de contaminar o treino.
"""

from __future__ import annotations

import hashlib
import io
import logging
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from creditwatch.data.schema import RAW_RENAME

logger = logging.getLogger("creditwatch.data.source")

UCI_URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
UCI_ZIP_SHA256 = "56c885f84457f6680f8438f02bfcdac9579323d8a94465ee5f26e32baa727602"
UCI_DOI = "10.24432/C55S3H"
# SHA-256 do CSV gerado a partir da planilha da UCI (snapshot versionado em data/raw).
RAW_CSV_SHA256 = "4edce6da569b2c30788a85fd98a9bafb1a983e8ca1a7aac875dd7ff853cbd0bb"
EXPECTED_ROWS = 30_000


class DatasetIntegrityError(RuntimeError):
    """O arquivo obtido nao confere com o hash esperado."""


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, timeout: int = 120) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "creditwatch/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - URL fixa (https)
        return response.read()


def fetch_raw_dataset(target: Path, url: str = UCI_URL, expected_sha256: str = UCI_ZIP_SHA256) -> Path:
    """Baixa o zip da UCI, confere o hash e grava o CSV normalizado em ``target``."""
    logger.info("baixando dataset de %s", url)
    payload = _download(url)
    digest = hashlib.sha256(payload).hexdigest()
    if digest != expected_sha256:
        raise DatasetIntegrityError(f"sha256 do download ({digest}) difere do esperado ({expected_sha256})")
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        name = next(n for n in archive.namelist() if n.lower().endswith(".xls"))
        frame = pd.read_excel(io.BytesIO(archive.read(name)), header=1)
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False, lineterminator="\n")
    logger.info("dataset bruto gravado em %s (%d linhas)", target, len(frame))
    return target


def load_raw_dataset(
    path: Path, download_if_missing: bool = True, expected_sha256: str | None = RAW_CSV_SHA256
) -> pd.DataFrame:
    """Le o dataset bruto, confere a integridade e normaliza os nomes das colunas (sem alterar valores).

    Alem do layout e do numero de linhas, o SHA-256 do CSV e comparado com o do snapshot
    versionado: um arquivo editado, mesmo com as 30.000 linhas, interrompe o pipeline.
    ``expected_sha256=None`` desliga essa conferencia (uso em testes com arquivos reduzidos).
    """
    path = Path(path)
    if not path.exists():
        if not download_if_missing:
            raise FileNotFoundError(path)
        fetch_raw_dataset(path)
    frame = pd.read_csv(path)
    missing = sorted(set(RAW_RENAME) - set(frame.columns))
    if missing:
        raise DatasetIntegrityError(f"colunas ausentes no arquivo bruto: {missing}")
    if len(frame) != EXPECTED_ROWS:
        raise DatasetIntegrityError(f"esperadas {EXPECTED_ROWS} linhas, encontradas {len(frame)}")
    if expected_sha256 is not None:
        digest = sha256_of(path)
        if digest != expected_sha256:
            raise DatasetIntegrityError(f"sha256 do CSV bruto ({digest}) difere do esperado ({expected_sha256})")
    return frame.rename(columns=RAW_RENAME)[list(RAW_RENAME.values())]
