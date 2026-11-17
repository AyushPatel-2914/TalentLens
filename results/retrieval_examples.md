# Retrieval examples (test split)

Three test jobs where BM25 and dense disagree most. Label: 2 strong, 1 partial, 0 no fit, ? unlabelled.

## J3900977839: Construction Project Coordinator NI Academy

- **bm25** nDCG@10 0.000: R19179079 (HR, label ?); R18297650 (HR, label ?); R57667857 (HR, label ?)
- **dense** nDCG@10 1.000: R17570634 (CONSTRUCTION, label 1); R30311725 (CONSTRUCTION, label 1); R12666174 (CONSTRUCTION, label 1)
- **hybrid** nDCG@10 0.632: R19179079 (HR, label ?); R39566718 (CONSTRUCTION, label 1); R27243670 (CONSTRUCTION, label 1)

## J3901390863: Sales Executive - Associate Staffing

- **bm25** nDCG@10 0.000: R10235211 (BUSINESS-DEVELOPMENT, label ?); R19179079 (HR, label ?); R17864043 (HEALTHCARE, label ?)
- **dense** nDCG@10 0.931: R34131484 (SALES, label 1); R69155584 (SALES, label 1); R25315791 (SALES, label 1)
- **hybrid** nDCG@10 0.095: R18311419 (BUSINESS-DEVELOPMENT, label ?); R47067533 (BUSINESS-DEVELOPMENT, label ?); R21856577 (BANKING, label ?)

## J3905333596: Commercial Carpenter

- **bm25** nDCG@10 0.073: R97169343 (HEALTHCARE, label ?); R11480899 (HR, label ?); R19179079 (HR, label ?)
- **dense** nDCG@10 0.936: R10100240 (CONSTRUCTION, label 1); R20565849 (CONSTRUCTION, label 1); R22718826 (CONSTRUCTION, label 1)
- **hybrid** nDCG@10 0.655: R10100240 (CONSTRUCTION, label 1); R29221006 (AVIATION, label ?); R39027764 (CONSTRUCTION, label 1)

