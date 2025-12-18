# Data Collection Workflow

This document records the provenance of `data/final1.csv`, detailing the scraping and cleaning steps used to build the assessment catalog.

## 1. Automated Scraping (`data_scrap/scrappers.ipynb`)
We utilized a Python-based automation script to crawl the SHL Product Catalog.

### Phase 1: Catalog Discovery
- **Target**: `https://www.shl.com/products/product-catalog/`
- **Method**: The script programmatically iterated through the catalog's dynamic pagination (`?start=0, 12, ...`).
- **Data Extracted**: Assessment names and primary product handles.

### Phase 2: Deep Link Extraction
For every discovered assessment, the script performed a targeted HTTP request to the specific product page to extract:
- **Long-form Description**: To enable detailed semantic search.
- **Job Levels**: (Entry, Mid, Senior) used for filtering.
- **Test Categories**: Key labels like "Ability & Aptitude" or "Personality".
- **Average Duration**: Time-to-complete metrics.

## 2. URL Canonicalization (`data_scrap/canonicalize_url.ipynb`)
To maintain compatibility with current production routing, the scraped URLs underwent a transformation:
- **Transformation**: String replacement of legacy product paths with current solutions paths.
- **Output**: `data/final1.csv` containing 377 verified assessment entries.

---
*Developed for internal data traceability. The current implementation uses the processed output directly.*
