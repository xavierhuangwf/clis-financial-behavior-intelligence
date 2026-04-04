# CLIS: Financial Behavior Intelligence

A modular analytics framework for behavioural signal analysis, transaction intelligence, customer segmentation, and intervention-oriented decision support in regulated financial environments.

![Overall Workflow](./docs/Overall_workflow.png)

## Overview

This repository presents a refactored implementation of the Customer Loyalty Improvement System (CLIS), originally developed as a modular analytics pipeline for transforming transaction data into actionable behavioural insights. In a financial-services setting, the system supports data cleaning, spending analysis, expenditure categorisation, customer-risk modelling, segmentation, and cashback-oriented recommendation design.

Although demonstrated on transaction data, the broader value of the repository lies in its reusable workflow patterns: structured preprocessing of sequential records, interpretable behavioural feature construction, modular machine learning pipelines, and decision-support outputs in regulated, data-intensive environments.

## Why This Matters

Modern institutions operating in regulated settings must convert large volumes of behavioural and operational data into timely, interpretable, and actionable signals. In the CLIS setting, these signals help identify spending patterns, detect behavioural shifts, and support targeted customer interventions. More broadly, the repository illustrates how practical machine learning systems can be organised for monitoring-oriented analytics, early signal extraction, and downstream decision support.

## Current Workflow

The refactored workflow currently follows this sequence:

1. **Preprocessing**
   - Validate schema and required columns
   - Remove invalid, duplicate, or incomplete records
   - Standardise account identifiers and timestamps
   - Construct a unified `Datetime` field
   - Extract consumer-initiated spending transactions

2. **Exploratory Data Analysis**
   - Generate paper-style descriptive figures
   - Summarise spending by category
   - Summarise transaction frequency by category
   - Visualise the filtered transaction amount distribution

3. **Transaction Categorisation**
   - Apply merchant-to-category mapping rules to construct labelled expenditure data
   - Build a categorized transaction dataset for downstream use
   - Provide an optional Random Forest categorization experiment using merchant, amount, balance, and temporal features

4. **Downstream Modules**
   - Churn modelling
   - Customer segmentation
   - Recommendation and intervention logic

The repository is being cleaned so that each stage becomes a clearer, more reproducible module in the overall pipeline.

## Current Implemented Components

### 1. Data Preprocessing
The preprocessing module handles core validation and cleaning operations, including:
- required-column checks
- date, time, and account-format validation
- null and duplicate removal
- account normalisation
- chronological sorting
- consumer-spending extraction

### 2. Exploratory Analysis
The EDA module produces descriptive outputs aligned with the project write-up, including:
- total spending by category
- transaction frequency by category
- filtered transaction amount distribution

### 3. Transaction Categorisation
The categorization module currently includes:
- a reusable merchant-to-category mapping table
- categorized dataset preparation
- an optional supervised categorization experiment using Random Forest

The rule-based mapping is the primary source of expenditure labels. The model-based categorization stage should be understood as a supplementary learned approximation of those merchant-derived labels rather than as a fully independent ground-truth categorization system.

## Repository Structure

- `docs/` workflow figures and supporting project notes
- `data/raw/` raw input data placeholders
- `data/processed/` processed and categorized transaction tables
- `notebooks/` exploratory notebooks retained for reference
- `outputs/figures/` generated visualisations
- `outputs/metrics/` evaluation reports
- `outputs/predictions/` predicted outputs from downstream modules
- `src/clis/data/` preprocessing and EDA
- `src/clis/categorization/` category mapping, labeled dataset preparation, and optional RF categorization
- `src/clis/churn/` churn-related components under refactoring
- `src/clis/segmentation/` segmentation-related components under refactoring
- `src/clis/recommendation/` recommendation-related components under refactoring
- `scripts/` helper scripts and notebook exports
- `reports/` report materials and supporting writeups

## Responsible Use

This repository is intended as a research and engineering prototype. Any real-world deployment should account for privacy, data governance, explainability, fairness, auditability, and appropriate human oversight. Predictive and behavioural outputs should not be used in isolation for consequential decision-making.

## Methodological Relevance

This repository demonstrates transferable technical capability in:
- structured preprocessing of raw sequential transaction records
- behavioural feature extraction
- interpretable descriptive analytics
- modular machine learning workflow design
- decision-support-oriented output generation

While implemented in a financial-services context, these workflow patterns are relevant to broader monitoring, early risk-identification, and decision-support tasks across other data-intensive and regulated domains.

## Notes

- Raw proprietary or sensitive data are not included in this repository.
- Large trained model binaries should not be version-controlled in GitHub.
- The repository should be understood as a modular applied machine learning prototype and refactoring effort rather than a production deployment.
