"""Tests for the region and country registry in momp.onset.regions."""
import pytest

from momp.onset import definition as dfn
from momp.onset import regions as rg

VALID = """\
id: testland
title: Testland
kind: country
iso: TST
"""


@pytest.fixture
def region_dir(tmp_path, monkeypatch):
    """Point the region registry at a temporary directory and return it."""
    monkeypatch.setattr(rg, "REGIONS_DIR", tmp_path)
    rg._index.cache_clear()
    yield tmp_path
    rg._index.cache_clear()


def write(directory, text, name="testland"):
    """Write a region file and return the load error it causes, if any."""
    (directory / f"{name}.yaml").write_text(text)
    rg._index.cache_clear()


def test_every_entry_loads():
    """All bundled regions parse and have consistent ids."""
    for ident in rg.region_ids():
        region = rg.load_region(ident)
        assert region.id == ident
        for parent in region.part_of:
            assert rg.load_region(parent)
    assert {"east_africa", "ethiopia", "kenya", "south_asia", "india", "west_africa", "sahel", "africa",
            "amazon"} <= set(rg.region_ids())


def test_aliases_and_case():
    """Names match by id or alias, ignoring case."""
    assert rg.load_region("Ethiopia").id == "ethiopia"
    assert rg.load_region("INDIA").iso == "IND"
    assert rg.load_region("East Africa").id == "east_africa"
    with pytest.raises(rg.RegionError, match="unknown region"):
        rg.load_region("atlantis")


def test_entry_contents():
    """Spot-check the data of two countries."""
    eth = rg.load_region("ethiopia")
    assert (eth.lat, eth.lon) == ((3.0, 15.0), (33.0, 48.0))
    assert (eth.observations_dataset, eth.day_boundary) == ("CHIRPS v3", "00 UTC, labelled by start")
    assert eth.climatology_years == (1998, 2024)
    ind = rg.load_region("india")
    assert ind.day_boundary == "03 UTC, labelled by end" and ind.climatology_years == (1991, 2020)


def test_ancestors():
    """Ancestors run nearest first."""
    assert [r.id for r in rg.ancestors(rg.load_region("sahel"))] == ["west_africa", "africa"]
    assert [r.id for r in rg.ancestors(rg.load_region("kenya"))] == ["east_africa", "africa"]
    assert rg.ancestors(rg.load_region("africa")) == []


def test_ancestors_cycle_safe(region_dir):
    """A part_of cycle terminates."""
    write(region_dir, "id: a\ntitle: A\nkind: region\npart_of: [b]\n", "a")
    write(region_dir, "id: b\ntitle: B\nkind: region\npart_of: [a]\n", "b")
    assert [r.id for r in rg.ancestors(rg.load_region("a"))] == ["b"]


def test_definitions_for_inheritance():
    """The nearest default wins and 'also' is the union along the chain."""
    assert rg.definitions_for("kenya") == ("icpac", ["dunning", "liebmann_marengo"])
    assert rg.definitions_for("ethiopia") == ("icpac", ["sivakumar", "dunning", "liebmann_marengo"])
    assert rg.definitions_for("sahel") == ("marteau", ["sivakumar", "omotosho", "dunning"])
    assert rg.definitions_for("india")[0] == "moron_robertson_mok"
    assert rg.definitions_for("south_asia") == (None, [])
    assert rg.definitions_for("africa") == (None, ["dunning"])


def test_check_definition_region():
    """Warn only when neither the area nor an ancestor is listed."""
    warning = rg.check_definition_region(dfn.load("moron_robertson"), "Ethiopia")
    assert "moron_robertson" in warning and "india" in warning and "Ethiopia" in warning
    assert rg.check_definition_region(dfn.load("icpac"), "kenya") is None
    assert rg.check_definition_region(dfn.load("dunning"), "kenya") is None
    assert rg.check_definition_region(dfn.load("moron_robertson"), "india") is None
    bare = dfn.loads(dfn.load("icpac").source_yaml.replace("regions:\n- east_africa\n", ""))
    assert bare.regions == () and rg.check_definition_region(bare, "india") is None


@pytest.mark.parametrize("extra, where, fix", [
    ("bogus: 1\n", "bogus", "unknown key"),
    ("iso: TS\n", "iso", "alpha-3"),
    ("kind: place\n", "kind", "'country' or 'region'"),
    ("part_of: [nowhere]\n", "part_of", "existing region"),
    ("domain: {lat: [3, 15]}\n", "domain", "missing 'lon'"),
    ("domain: {lat: [15, 3], lon: [1, 2]}\n", "domain.lat", "low < high"),
    ("observations: {dataset: X, day_boundary: noon}\n", "observations.day_boundary", "Fix:"),
    ("observations: {dataset: X}\n", "observations.day_boundary", "required"),
    ("climatology: {years: 2000-1990}\n", "climatology.years", "YYYY-YYYY"),
    ("climatology: {years: 1990}\n", "climatology.years", "YYYY-YYYY"),
    ("definitions: {default: nope}\n", "definitions.default", "registry"),
    ("definitions: {also: [icpac, nope]}\n", "definitions.also[1]", "registry"),
    ("definitions: {defaults: icpac}\n", "definitions.defaults", "unknown key"),
    ("aliases: Testland\n", "aliases", "list"),
])
def test_validation_errors(region_dir, extra, where, fix):
    """Each error names the YAML path and says how to fix it."""
    text = VALID.replace("kind: country\niso: TST\n", "") + ("kind: country\niso: TST\n") + extra
    if extra.startswith("kind:"):
        text = "id: testland\ntitle: Testland\n" + extra
    if extra.startswith("iso:"):
        text = "id: testland\ntitle: Testland\nkind: country\n" + extra
    write(region_dir, text)
    with pytest.raises(rg.RegionError) as exc:
        rg.load_region("testland")
    assert where in str(exc.value) and fix in str(exc.value), str(exc.value)


def test_structural_errors(region_dir):
    """Missing keys, region with iso, id mismatch and duplicate aliases."""
    write(region_dir, "id: testland\nkind: region\n")
    with pytest.raises(rg.RegionError, match="missing required key 'title'"):
        rg.load_region("testland")
    write(region_dir, "id: testland\ntitle: T\nkind: region\niso: TST\n")
    with pytest.raises(rg.RegionError, match="only for countries"):
        rg.load_region("testland")
    write(region_dir, "id: other\ntitle: T\nkind: region\n")
    with pytest.raises(rg.RegionError, match="file name"):
        rg.load_region("other")
    write(region_dir, "id: testland\ntitle: T\nkind: region\naliases: [Shared]\n")
    write(region_dir, "id: second\ntitle: T\nkind: region\naliases: [shared]\n", "second")
    with pytest.raises(rg.RegionError, match="already used"):
        rg.load_region("second")


def test_definition_regions_key():
    """A definition's regions are validated and carried, but not hashed."""
    text = dfn.load("icpac").source_yaml
    d = dfn.loads(text)
    assert d.regions == ("east_africa",)
    other = dfn.loads(text.replace("- east_africa", "- sahel"))
    assert other.regions == ("sahel",) and dfn.definition_hash(other) == dfn.definition_hash(d)
    with pytest.raises(dfn.DefinitionError) as exc:
        dfn.loads(text.replace("- east_africa", "- narnia"))
    assert "regions[0]" in str(exc.value) and "Fix:" in str(exc.value)
    with pytest.raises(dfn.DefinitionError, match="regions"):
        dfn.loads(text.replace("regions:\n- east_africa", "regions: east_africa"))
