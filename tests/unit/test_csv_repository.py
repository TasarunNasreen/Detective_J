from pathlib import Path

from multimodal_evidence_review.infrastructure.csv_repository import CsvRepository, parse_list


def test_parse_list_supports_json_and_delimiters() -> None:
    assert parse_list('["a.jpg", "b.png"]') == ["a.jpg", "b.png"]
    assert parse_list("a.jpg;b.png") == ["a.jpg", "b.png"]


def test_load_claims_accepts_documented_schema(tmp_path: Path) -> None:
    path = tmp_path / "claims.csv"
    path.write_text(
        'claim_id,user_id,user_claim,image_ids\nC1,U1,Dented door,"[""a.jpg"", ""b.jpg""]"\n',
        encoding="utf-8",
    )
    claims, _ = CsvRepository().load_claims(path)
    assert claims[0].claim_id == "C1"
    assert claims[0].image_ids == ["a.jpg", "b.jpg"]

