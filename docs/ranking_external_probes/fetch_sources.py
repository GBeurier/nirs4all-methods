"""Download immutable upstream sources into a disposable probe directory."""
import io
import pathlib
import tarfile
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent
SOURCES = {
    "lme4": "https://codeload.github.com/lme4/lme4/tar.gz/48e197dca2fc2e74870010fcfedea55ffc9dace9",
    "rfr": "https://codeload.github.com/automl/random_forest_run/tar.gz/1ba17650e734edd3ea536efc91d32815c4874e72",
    "mixeff": "https://crates.io/api/v1/crates/mixeff-rs/1.0.0-rc.5/download",
    "glmm": "https://crates.io/api/v1/crates/glmm/0.8.0/download",
    "openblup": "https://codeload.github.com/jakobrichert/openblup/tar.gz/0dd1f37ffe7f7caf9fda51ecdd4a24eeba7b2ce0",
}
for name, url in SOURCES.items():
    destination = ROOT / name
    destination.mkdir(exist_ok=True)
    with urllib.request.urlopen(url) as response:
        archive = response.read()
    with tarfile.open(fileobj=io.BytesIO(archive)) as source:
        for member in source.getmembers():
            parts = member.name.split("/", 1)
            if len(parts) == 2 and parts[1]:
                member.name = parts[1]
                source.extract(member, destination, filter="data")
    print(f"{name}: {url}")
