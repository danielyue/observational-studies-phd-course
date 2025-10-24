# HuggingFace Hub Stats Data Analysis Report

**Dataset**: `cfahlgren1/hub-stats` - models.parquet file
**Date**: October 23, 2025
**Total Models Analyzed**: 2,161,698

---

## Executive Summary

This report documents the representation of three key data concepts in the HuggingFace hub-stats models.parquet dataset:
1. **Downloads and Downloads Over Time** - How download metrics distinguish recent vs. historical activity
2. **Fine-tuning Information** - How model trees and parent-child relationships are represented
3. **Parameter Counts** - How model sizes are stored and categorized

---

## 1. Downloads: Recent vs. All-Time

### Key Fields
- **`downloads`**: Recent download count (last ~30 days)
- **`downloadsAllTime`**: Cumulative downloads since model creation

### Key Findings

#### Coverage
- Models with recent downloads (>0): **767,610** (35.5%)
- Models with all-time downloads (>0): **1,290,554** (59.7%)

#### Statistical Distribution
| Metric | Recent (`downloads`) | All-Time (`downloadsAllTime`) |
|--------|---------------------|-------------------------------|
| Median | 0 | 19 |
| Mean | 812 | 21,641 |

#### Important Insight
Recent downloads account for only **3.75%** of all-time downloads, indicating:
- Most models have accumulated downloads over long periods
- Recent activity is a small fraction of historical totals
- Many models experience declining usage over time

### Use Cases

**Use `downloads` (recent) when:**
- Tracking current popularity trends
- Identifying actively-used models
- Finding emerging popular models
- Analyzing month-over-month growth

**Use `downloadsAllTime` when:**
- Understanding historical impact
- Finding established/proven models
- Analyzing total reach and adoption
- Comparing long-term model success

### Example: Popular but Declining Model
```
Model: 1231czx/llama3_it_ultra_list_and_bold500
Recent downloads (30 days): 1
All-time downloads: 71,152,684
Ratio: 0.000014% (was popular, now rarely used)
```

### Example: Currently Active Model
```
Model: sentence-transformers/all-MiniLM-L6-v2
Recent downloads (30 days): 129,917,294
All-time downloads: 1,520,703,899
Ratio: 8.54% (consistently popular over time)
```

---

## 2. Fine-tuning and Model Trees

### Key Field: `baseModels`

#### Structure
```json
{
  "models": [
    {
      "_id": "internal_id",
      "id": "author/model-name"
    }
  ],
  "relation": "finetune" | "adapter" | "quantized" | "merge"
}
```

### Key Findings

#### Coverage
- Models with baseModels populated: **607,148** (28.09%)
- This means about **72%** of models do not have explicit parent relationships documented

#### Relationship Types Distribution

| Relation Type | Count | Percentage |
|--------------|-------|------------|
| adapter | 258,462 | 42.57% |
| finetune | 218,000 | 35.91% |
| quantized | 116,696 | 19.22% |
| merge | 13,990 | 2.30% |

### Understanding Relationship Types

1. **finetune**: Model was trained further on the base model with new data
   - Example: Taking a general model and training it for a specific task

2. **adapter**: Parameter-efficient fine-tuning (PEFT) approach
   - Adds small trainable layers while freezing base model
   - More efficient than full fine-tuning

3. **quantized**: Model weights converted to lower precision
   - Reduces model size and memory requirements
   - Examples: Converting FP32 → INT8, BF16 → 4-bit

4. **merge**: Combination of multiple models
   - Blends weights from different models
   - Used to combine capabilities

### Example: Qwen3-VL-8B-Instruct Model Tree

**Base Model**: Qwen/Qwen3-VL-8B-Instruct
- Downloads: 178,779
- Created: 2025-10-11
- Likes: 300

**Derivative Models** (23 found):

1. **Quantized Versions** (for efficiency):
   - `NexaAI/Qwen3-VL-8B-Instruct-GGUF` (22,391 downloads)
   - `unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit` (8,000 downloads)
   - `cpatonn/Qwen3-VL-8B-Instruct-AWQ-4bit` (3,260 downloads)

2. **Fine-tuned Versions** (for specific tasks):
   - `huihui-ai/Huihui-Qwen3-VL-8B-Instruct-abliterated` (587 downloads)
   - `prithivMLmods/Qwen3-VL-8B-Instruct-abliterated` (145 downloads)

### Accessing on HuggingFace Website

When viewing a model like https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct:

1. **Model Card Metadata**: The `base_model` field in the YAML frontmatter shows parent models
2. **"Community" Tab**: May show derivative models
3. **Model Tree Visualization**: Some models display a tree graph of relationships

The `baseModels` field in the dataset is the programmatic representation of this information.

---

## 3. Parameter Counts

### Key Field: `safetensors`

#### Structure
```json
{
  "parameters": {
    "BF16": count,    // BFloat16 parameters
    "F16": count,     // Float16 parameters
    "F32": count,     // Float32 parameters
    "F64": count,     // Float64 parameters
    "I8": count,      // Int8 (quantized)
    "U8": count,      // UInt8 (quantized)
    // ... other data types
  },
  "total": total_parameter_count
}
```

### Key Findings

#### Coverage
- Models with parameter information: **505,406** (23.38%)
- This means **76%** of models lack explicit parameter counts in the data

#### Parameter Count Distribution

| Percentile | Parameter Count (Billions) |
|-----------|---------------------------|
| Minimum | 0.0000B |
| 25th | 0.1244B (124M) |
| Median | 0.7645B (765M) |
| 75th | 6.7384B (6.7B) |
| Maximum | 684.53B |

### Example: Qwen3-VL-8B-Instruct

```
Total parameters: 8,767,123,696
                = 8.77 billion

Parameter breakdown by data type:
  BF16: 8,767,123,696 (8.77B)
```

This shows the model stores all parameters in **BFloat16** precision, a common choice for:
- Good balance between precision and memory
- Native support in modern GPUs
- Stable training and inference

### Understanding Data Types

| Type | Bits | Use Case |
|------|------|----------|
| F64 | 64 | Research/high-precision scientific |
| F32 | 32 | Traditional training |
| BF16 | 16 | Modern efficient training |
| F16 | 16 | Inference optimization |
| I8 | 8 | Quantization for deployment |
| I4 | 4 | Aggressive quantization |

**Lower precision = smaller model size, faster inference, but potential accuracy trade-offs**

### Most Downloaded Models by Size

#### Small Models (<1B parameters)
1. `google-bert/bert-base-uncased` - 0.11B - 2.5B downloads
2. `MIT/ast-finetuned-audioset-10-10-0.4593` - 0.09B - 2.2B downloads
3. `sentence-transformers/all-MiniLM-L6-v2` - 0.02B - 1.5B downloads

#### Medium Models (7-15B parameters)
1. `meta-llama/Llama-3.1-8B-Instruct` - 8.03B - 91M downloads
2. `1231czx/llama3_it_ultra_list_and_bold500` - 7.50B - 71M downloads
3. `Qwen/Qwen2.5-7B-Instruct` - 7.62B - 48M downloads

#### Large Models (100B+ parameters)
1. `meta-llama/Llama-3.1-405B` - 405.85B - 18M downloads
2. `deepseek-ai/DeepSeek-R1` - 684.53B - 11M downloads
3. `deepseek-ai/DeepSeek-V3` - 684.53B - 9M downloads

---

## Practical Applications

### 1. Analyzing Model Popularity Trends

```python
import pandas as pd

df = pd.read_parquet('models.parquet')

# Find models with growing popularity (high recent relative to total)
df['recent_ratio'] = df['downloads'] / df['downloadsAllTime']
trending = df[df['recent_ratio'] > 0.10]  # >10% downloads are recent
```

### 2. Mapping Model Families

```python
# Find all derivatives of a base model
def find_derivatives(base_model_id):
    derivatives = []
    for idx, row in df.iterrows():
        if row['baseModels']:
            for model in row['baseModels'].get('models', []):
                if model.get('id') == base_model_id:
                    derivatives.append({
                        'id': row['id'],
                        'relation': row['baseModels']['relation']
                    })
    return derivatives
```

### 3. Filtering by Model Size

```python
# Find models in a specific size range
def get_models_by_size(min_b, max_b):
    filtered = []
    for idx, row in df.iterrows():
        if row['safetensors']:
            params_b = row['safetensors']['total'] / 1e9
            if min_b <= params_b <= max_b:
                filtered.append(row)
    return filtered

# Get all 7B models
seven_b_models = get_models_by_size(6.5, 8.5)
```

---

## Data Quality Notes

### Missing Data Patterns

1. **baseModels**: 72% of models don't have this field
   - Older models less likely to have it
   - User-uploaded models often missing metadata
   - Not all fine-tuning relationships are documented

2. **safetensors**: 77% of models lack parameter counts
   - Models without safetensors format
   - Very old models
   - Non-transformer architectures

3. **downloads**: 40% of models have zero recent downloads
   - Abandoned/experimental models
   - Duplicate uploads
   - Private/testing models made public

### Recommendations for Analysis

1. **Always check for null/missing values** before analysis
2. **Filter by metadata completeness** for research requiring specific fields
3. **Cross-reference with HuggingFace API** for most up-to-date information
4. **Consider date filters** (`createdAt`, `lastModified`) to focus on recent models

---

## Exploratory Scripts Provided

Two analysis scripts have been created for further exploration:

1. **`explore_models_data.py`**: Initial data exploration
   - Schema inspection
   - Column enumeration
   - Sample data viewing

2. **`analyze_data_concepts.py`**: Deep-dive analysis
   - Downloads comparison
   - Model tree analysis
   - Parameter distribution analysis

Run with: `uv run analyze_data_concepts.py`

---

## References

- **Dataset**: https://huggingface.co/datasets/cfahlgren1/hub-stats
- **HuggingFace Hub API**: https://huggingface.co/docs/hub/api
- **Models Download Stats**: https://huggingface.co/docs/hub/models-download-stats
- **Example Model with Tree**: https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
