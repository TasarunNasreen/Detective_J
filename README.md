# Detective_J

Detective_J implements an image-first damage-claim review pipeline for the
HackerRank **Multi-Modal Evidence Review** challenge. It reads the four required
CSVs plus local images, calls the OpenAI Responses API with image input, applies
deterministic evidence and risk rules, and writes `output.csv`.

## 1. Project structure

```text
.
├── dataset/
│   ├── claims.csv
│   ├── sample_claims.csv
│   ├── user_history.csv
│   ├── evidence_requirements.csv
│   └── images/
├── evaluation/
│   └── evaluate.py
├── src/multimodal_evidence_review/
│   ├── application/       # Review orchestration
│   ├── domain/            # Typed models and enums
│   ├── infrastructure/    # CSV, images, and Responses API
│   └── validation/        # Evidence, risk, and output rules
├── tests/
├── main.py
├── pyproject.toml
└── requirements.txt
```

## 2. Install

Python 3.11 or newer is required.

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

### macOS/Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## 3. Add the dataset

Copy the challenge files into `dataset/` using their exact names:

- `claims.csv`
- `sample_claims.csv`
- `user_history.csv`
- `evidence_requirements.csv`
- image files under `dataset/images/`

`claims.csv` and `sample_claims.csv` must contain `claim_id` and `user_claim`.
The loader also accepts common aliases documented in
`infrastructure/csv_repository.py`. `user_id` is needed to attach history.
Images may be listed in an `image_ids` JSON array or delimiter-separated field,
in numbered `image_1`, `image_2`, etc. fields, or named with a claim-ID prefix.

`user_history.csv` must contain `user_id`. Other history columns are interpreted
only by the risk module. They are never sent to the OpenAI model.

`evidence_requirements.csv` must contain `issue_type`. Supported rule columns are:

- `object_part` and `severity` for more specific matching;
- `min_images` (default `1`);
- `min_quality` (`unusable`, `poor`, `fair`, `good`, or `excellent`);
- `required_views`, represented as JSON or separated with `;`, `,`, or `|`.

## 4. Configure OpenAI

Copy `.env.example` to `.env`, then set `OPENAI_API_KEY`. The default model is
`gpt-4.1-mini`; override it with `OPENAI_MODEL` or `--model`. The configured model
must accept image input and structured outputs through the Responses API.

The SDK call is isolated in `infrastructure/openai_vision.py`. It uses
`client.responses.parse(...)` with a strict Pydantic schema and sends each local
image as a base64 data URL. See the official [OpenAI vision guide](https://developers.openai.com/api/docs/guides/images-vision)
and [Responses API reference](https://developers.openai.com/api/reference/responses).

## 5. Generate output.csv

After activating the environment:

```powershell
python main.py --dataset-dir dataset --output output.csv
```

or, after editable installation:

```powershell
evidence-review --dataset-dir dataset --output output.csv
```

The repository-branded command is also available:

```powershell
detective-j --dataset-dir dataset --output output.csv
```

The output schema is enforced in this exact order:

```text
claim_id,damage_claim,classification,issue_type,object_part,severity,image_quality,supporting_image_ids,risk_flags,evidence_standard_met,justification
```

`classification` is exactly one of `supported`, `contradicted`, or
`not_enough_information`. List columns are JSON arrays and the evidence boolean is
written as lowercase `true` or `false`.

## 6. How decisions are made

1. Pillow verifies and normalizes EXIF orientation for each image.
2. OpenCV measures resolution, blur, and brightness before upload.
3. The Responses API extracts the damage allegation and reviews every image.
4. Model-returned image IDs are restricted to IDs actually supplied.
5. `evidence_requirements.csv` is matched by issue, part, and severity. A decisive
   result is downgraded to `not_enough_information` if minimum evidence fails.
6. History is processed afterward to produce only the allowed risk flags. It never
   changes classification, severity, or visible observations.
7. A strict writer validates flags, duplicate claim IDs, booleans, JSON lists, and
   column order before creating `output.csv`.

Allowed risk flags are `adverse_claim_history`, `duplicate_evidence_history`,
`high_claim_frequency`, `new_account`, `prior_fraud_indicator`, and
`repeated_similar_claims`. No flags are emitted when history supplies no supported
risk signal.

## 7. Run evaluation

```powershell
python -m evaluation.evaluate --dataset-dir dataset
```

This reviews `sample_claims.csv` and writes:

- `evaluation/evaluation_results.csv`
- `evaluation/evaluation_report.md`

Accuracy is calculated when the sample file contains supported expected or
ground-truth columns. The report always records runtime, API latency, and measured
token usage. For a USD cost estimate, set the two current model prices:

```text
OPENAI_INPUT_COST_PER_1M_TOKENS
OPENAI_OUTPUT_COST_PER_1M_TOKENS
```

Pricing is deliberately configuration, not hardcoded data, because API pricing can
change.

## 8. Test

```powershell
pytest
```

Tests cover input parsing, evidence thresholds, history-only risk flags, exact
output schema, and the decisive-to-insufficient downgrade. The integration test
uses a fake analyzer, so it does not spend API credits.

## Operational notes

- Logs are written to `logs/evidence_review.log` with claim IDs and rotation.
- API calls retry transient connection, timeout, rate-limit, and server failures.
- Images are resized before upload while preserving aspect ratio.
- Missing or unreadable evidence produces `not_enough_information`; it never causes
  a fabricated positive or negative decision.
- Do not commit `.env`, uploaded evidence, logs, or generated outputs.
