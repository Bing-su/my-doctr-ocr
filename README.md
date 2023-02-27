# my-doctr-ocr

## Inference

```python
# 불러오기
from doctr.io import DocumentFile
from doctr.models import ocr_predictor, from_hub

# opencv를 쓰기 때문에 영어 이외의 문자가 경로에 들어가면 안됨
img = DocumentFile.from_images(["image.jpg"])

det = from_hub("smartmind/doctr-db_resnet50")

rec = from_hub("smartmind/doctr-vitstr_small-recognition")
# 또는
# rec = from_hub("smartmind/doctr-vitstr_base-recognition")

model = ocr_predictor(det_arch=det, reco_arch=rec)
```

```python
# 추론

result = model(img)
result.show(img)
```

```python
# json으로 결과 보기
>>> result
Output exceeds the size limit. Open the full output data in a text editor
Document(
  (pages): [Page(
    dimensions=(905, 640)
    (blocks): [
      Block(
        (lines): [
          Line(
            (words): [
              Word(value='상의적', confidence=0.22),
              Word(value='인주시면을', confidence=0.66),
              Word(value='이르는', confidence=0.27),
              Word(value='배ᄉᄉᆫ이ᅵ', confidence=0.16),
            ]
          ),
          Line(
            (words): [Word(value='서울특별시교육청', confidence=0.4)]
          ),
        ]
        (artefacts): []
      ),
      Block(
        (lines): [
          Line(
            (words): [Word(value='88', confidence=0.21)]
          ),
...
        (artefacts): []
      ),
    ]
  )]
```

```python
# 모든 한국어는 nfd로 변환되어 있으므로(자모 분리 상태) 필요하면 nfc로 변환해야 함
from unicodedata import normalize

for page in result:
    for block in page:
        for line in block:
            for word in line:
                print(normalize("NFC", word.value))
```