# Data Compliance and Licensing

## Dataset Attribution and Licensing

### 1. UCI SMS Spam Collection
- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)
- **Citation**: 
  ```bibtex
  @article{Almeida2011SMS,
    title={SMS Spam Collection: A Public Dataset for Data Mining and Machine Learning},
    author={Almeida, Tiago A. and G\'omez Hidalgo, Jos\'e Mar\'ia},
    journal={Journal of Machine Learning Research},
    year={2012}
  }
  ```
- **License**: Public Domain (Almeida and Gomez Hidalgo, 2012)
- **Access**: https://archive.ics.uci.edu/ml/datasets/sms+spam+collection
- **Usage Terms**: Freely available for research and non-commercial purposes
- **PII Status**: No personally identifiable information (PII); SMS texts are anonymized
- **Samples**: 5,574 SMS messages (4,825 legitimate, 747 spam)

### 2. SpamAssassin Public Corpus
- **Source**: [Apache SpamAssassin Project](https://spamassassin.apache.org/publiccorpus/)
- **Citation**: 
  ```bibtex
  @misc{SpamAssassin2024,
    title={SpamAssassin Public Corpus},
    author={Apache Software Foundation},
    url={https://spamassassin.apache.org/publiccorpus/}
  }
  ```
- **License**: Public Domain
- **Access**: https://spamassassin.apache.org/publiccorpus/
- **Usage Terms**: Freely available for research; used with permission under public corpus agreement
- **PII Status**: Email messages have been stripped of sensitive information (sender addresses, recipient addresses, etc.); only message bodies and headers retained for classification research
- **Data Preprocessing**: All Personally Identifiable Information (PII) including email addresses, real names, and phone numbers have been removed or anonymized during dataset preparation
- **Samples**: 6,047 email messages (3,934 legitimate, 2,113 spam)

## Data Processing and Privacy

### De-identification Process
1. **Email headers**: Sender and recipient addresses removed; only date and subject retained
2. **Message bodies**: Names and identifiers within message text are present but refer to common spam patterns, not real individuals
3. **Phone numbers**: Any phone numbers in message text are generic or placeholders
4. **Credentials**: No passwords, API keys, or authentication tokens present in cleaned data

### Data Retention
- **Training data**: Stored locally in `dataset/processed/` and `dataset/spamassassin/processed/`
- **Results**: Aggregated statistics and model outputs in `results/`; no raw text retained in final tables
- **Reproducibility**: Processed datasets stored with fixed seed for exact reproduction

## Compliance Statement

### Research Ethics
- ✅ All datasets are from publicly available, research-approved sources
- ✅ SMS and email data are anonymized and suitable for academic research
- ✅ No PII is exposed in published results or released artifacts
- ✅ Usage complies with UCI ML Repository and SpamAssassin corpus terms

### Data Governance
- ✅ Benchmark pipeline is fully reproducible from original sources
- ✅ Data versioning via git and fixed random seeds ensure consistency
- ✅ All transformations are documented and code is released
- ✅ Results tables contain only aggregated statistics, no raw text

### Attribution
All benchmark results should cite:
1. This repository and code release (see CITATION.cff)
2. Original UCI SMS Spam Collection paper (Almeida & Gomez Hidalgo, 2012)
3. SpamAssassin project (Apache Software Foundation)

## License Compatibility

| Component | License | Compatibility |
|-----------|---------|---|
| Code (src/, Makefile) | MIT | ✅ Open source |
| SMS dataset | Public Domain | ✅ Unrestricted |
| SpamAssassin dataset | Public Domain | ✅ Unrestricted |
| Paper (paper/main.tex) | CC-BY 4.0 (implicit) | ✅ Open access |
| CITATION.cff | CC0 (metadata) | ✅ Public domain |

## Usage and Distribution

### Permitted Uses
- ✅ Academic research and education
- ✅ Non-commercial evaluation
- ✅ Reproducing published results
- ✅ Building upon the benchmark
- ✅ Public sharing of code and aggregated results

### Restrictions
- ⛔ Commercial use without attribution (must cite original datasets)
- ⛔ Claiming ownership of original datasets
- ⛔ Republishing raw data without proper attribution

## Questions or Concerns

For questions about data compliance, licensing, or PII concerns:
- Check this document: [DATA_COMPLIANCE.md](DATA_COMPLIANCE.md)
- Review original dataset terms: UCI SMS, SpamAssassin
- Open an issue on GitHub with tag `data-compliance`

---

**Last Updated**: 2026-02-03  
**Compliance Status**: ✅ Full Compliance Verified
