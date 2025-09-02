from pydantic import DirectoryPath

from shared.const import EXO_HOME

def build_model_path(model_id: str) -> DirectoryPath:
    return EXO_HOME / "models" / model_id.replace("/", "--")
