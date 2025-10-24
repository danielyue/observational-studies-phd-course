"""
Deep analysis of specific data concepts in models.parquet:
1. Downloads vs downloadsAllTime
2. Finetuning/model tree (baseModels)
3. Parameter counts (safetensors)
"""
import pandas as pd
import json
from pathlib import Path

# Load the data
data_path = Path(__file__).parent.parent / 'data' / 'raw' / 'models.parquet'
df = pd.read_parquet(data_path)

print("="*80)
print("ANALYSIS 1: DOWNLOADS vs DOWNLOADSALLTIME")
print("="*80)

print("\n1.1 Basic Statistics:")
print("-" * 80)
print(f"Total models: {len(df):,}")
print(f"\nModels with downloads > 0: {(df['downloads'] > 0).sum():,}")
print(f"Models with downloadsAllTime > 0: {(df['downloadsAllTime'] > 0).sum():,}")

print("\n1.2 Understanding the Difference:")
print("-" * 80)
# Calculate the ratio
df['download_ratio'] = df['downloads'] / df['downloadsAllTime'].replace(0, 1)
df['download_difference'] = df['downloadsAllTime'] - df['downloads']

print("\nKey insight: downloads represents RECENT downloads (last ~30 days)")
print("downloadsAllTime represents TOTAL downloads since model creation")
print("\nExamples showing the difference:")

# Find models with significant history (old downloads but few recent)
old_but_inactive = df[
    (df['downloadsAllTime'] > 10000) &
    (df['downloads'] < 100) &
    (df['downloads'] > 0)
].sort_values('downloadsAllTime', ascending=False).head(5)

print("\n  Models with high all-time downloads but low recent downloads:")
print("  (Models that were popular in the past but less used now)")
for idx, row in old_but_inactive.iterrows():
    print(f"    - {row['id']}")
    print(f"      Recent downloads (30 days): {row['downloads']:,}")
    print(f"      All-time downloads: {row['downloadsAllTime']:,}")
    print(f"      Ratio (recent/total): {row['download_ratio']:.4f}")
    print()

# Find models with high recent activity
recent_popular = df[
    (df['downloads'] > 100000)
].sort_values('downloads', ascending=False).head(5)

print("\n  Models with very high recent download activity:")
for idx, row in recent_popular.iterrows():
    print(f"    - {row['id']}")
    print(f"      Recent downloads (30 days): {row['downloads']:,}")
    print(f"      All-time downloads: {row['downloadsAllTime']:,}")
    created_date = row['createdAt'].strftime('%Y-%m-%d') if pd.notna(row['createdAt']) else 'Unknown'
    print(f"      Created: {created_date}")
    print()

print("\n1.3 Summary Statistics:")
print("-" * 80)
print(f"Median recent downloads: {df['downloads'].median():,.0f}")
print(f"Median all-time downloads: {df['downloadsAllTime'].median():,.0f}")
print(f"Mean recent downloads: {df['downloads'].mean():,.0f}")
print(f"Mean all-time downloads: {df['downloadsAllTime'].mean():,.0f}")

# Calculate what percentage of all-time downloads happened recently
total_recent = df['downloads'].sum()
total_alltime = df['downloadsAllTime'].sum()
print(f"\nRecent downloads as % of all-time: {(total_recent/total_alltime*100):.2f}%")

print("\n" + "="*80)
print("ANALYSIS 2: FINETUNING AND MODEL TREES (baseModels)")
print("="*80)

print("\n2.1 BaseModels Structure:")
print("-" * 80)

# Count models with baseModels
models_with_base = df['baseModels'].notna().sum()
print(f"Models with baseModels field populated: {models_with_base:,}")
print(f"Percentage: {(models_with_base/len(df)*100):.2f}%")

print("\n2.2 Understanding the baseModels Field:")
print("-" * 80)
print("The baseModels field is a dictionary with structure:")
print("  {")
print("    'models': [{'_id': '...', 'id': 'author/model-name'}],")
print("    'relation': 'finetune' | 'adapter' | 'merge' | etc.")
print("  }")
print("\nThis creates a model tree/graph where:")
print("  - Nodes are models")
print("  - Edges represent relationships (fine-tuning, adapters, merges)")

print("\n2.3 Examples of Model Trees:")
print("-" * 80)

# Find models with baseModels
models_with_base_data = df[df['baseModels'].notna()].copy()

print("\nExample 1: Qwen3-VL-8B-Instruct and its derivatives")
qwen_base = df[df['id'] == 'Qwen/Qwen3-VL-8B-Instruct'].iloc[0]
print(f"\nBase Model: {qwen_base['id']}")
print(f"  All-time downloads: {qwen_base['downloadsAllTime']:,}")
print(f"  Created: {qwen_base['createdAt'].strftime('%Y-%m-%d')}")
print(f"  Likes: {qwen_base['likes']}")

# Find models that have Qwen3-VL-8B-Instruct as base
qwen_derivatives = []
for idx, row in models_with_base_data.iterrows():
    if row['baseModels'] and isinstance(row['baseModels'], dict):
        models_list = row['baseModels'].get('models', [])
        if models_list is not None and len(models_list) > 0:
            for base_model in models_list:
                if isinstance(base_model, dict) and base_model.get('id') == 'Qwen/Qwen3-VL-8B-Instruct':
                    qwen_derivatives.append({
                        'id': row['id'],
                        'relation': row['baseModels'].get('relation', 'unknown'),
                        'downloads': row['downloadsAllTime'],
                        'created': row['createdAt']
                    })

print(f"\nFound {len(qwen_derivatives)} models derived from Qwen3-VL-8B-Instruct:")
for i, deriv in enumerate(qwen_derivatives[:10], 1):  # Show first 10
    print(f"  {i}. {deriv['id']}")
    print(f"     Relation: {deriv['relation']}")
    print(f"     Downloads: {deriv['downloads']:,}")

print("\n2.4 Relation Types Distribution:")
print("-" * 80)
relation_types = {}
for idx, row in models_with_base_data.iterrows():
    if row['baseModels'] and isinstance(row['baseModels'], dict):
        relation = row['baseModels'].get('relation', 'unknown')
        relation_types[relation] = relation_types.get(relation, 0) + 1

print("Distribution of relationship types:")
for relation, count in sorted(relation_types.items(), key=lambda x: x[1], reverse=True):
    print(f"  {relation}: {count:,} models ({count/models_with_base*100:.2f}%)")

print("\n" + "="*80)
print("ANALYSIS 3: PARAMETER COUNTS (safetensors)")
print("="*80)

print("\n3.1 SafeTensors Structure:")
print("-" * 80)
print("The safetensors field contains parameter information with structure:")
print("  {")
print("    'parameters': {")
print("      'BF16': count,    # BFloat16 parameters")
print("      'F16': count,     # Float16 parameters")
print("      'F32': count,     # Float32 parameters")
print("      'I8': count,      # Int8 parameters (quantization)")
print("      ... (other data types)")
print("    },")
print("    'total': total_parameter_count")
print("  }")

models_with_params = df['safetensors'].notna().sum()
print(f"\nModels with parameter information: {models_with_params:,}")
print(f"Percentage: {(models_with_params/len(df)*100):.2f}%")

print("\n3.2 Extracting Parameter Counts:")
print("-" * 80)

# Extract total parameters
param_counts = []
for idx, row in df[df['safetensors'].notna()].iterrows():
    if isinstance(row['safetensors'], dict):
        total = row['safetensors'].get('total')
        if total and total > 0:
            param_counts.append({
                'id': row['id'],
                'params': total,
                'params_billions': total / 1e9,
                'downloads': row['downloadsAllTime'],
                'likes': row['likes']
            })

param_df = pd.DataFrame(param_counts)

if len(param_df) > 0:
    print(f"Models with valid parameter counts: {len(param_df):,}")
    print(f"\nParameter count statistics (in billions):")
    print(f"  Min: {param_df['params_billions'].min():.4f}B")
    print(f"  25th percentile: {param_df['params_billions'].quantile(0.25):.4f}B")
    print(f"  Median: {param_df['params_billions'].median():.4f}B")
    print(f"  75th percentile: {param_df['params_billions'].quantile(0.75):.4f}B")
    print(f"  Max: {param_df['params_billions'].max():.2f}B")

    print("\n3.3 Example: Qwen3-VL-8B-Instruct Parameter Details:")
    print("-" * 80)
    qwen_params = df[df['id'] == 'Qwen/Qwen3-VL-8B-Instruct'].iloc[0]['safetensors']
    if qwen_params:
        print(f"Total parameters: {qwen_params['total']:,.0f}")
        print(f"                 = {qwen_params['total']/1e9:.2f} billion")
        print("\nParameter breakdown by data type:")
        if 'parameters' in qwen_params:
            for dtype, count in qwen_params['parameters'].items():
                if count and count > 0:
                    print(f"  {dtype}: {count:,.0f} ({count/1e9:.2f}B)")

    print("\n3.4 Most Popular Models by Parameter Size:")
    print("-" * 80)

    # Group by size ranges
    param_df['size_category'] = pd.cut(
        param_df['params_billions'],
        bins=[0, 1, 3, 7, 15, 30, 100, float('inf')],
        labels=['<1B', '1-3B', '3-7B', '7-15B', '15-30B', '30-100B', '100B+']
    )

    print("\nTop 3 most downloaded models in each size category:")
    for category in ['<1B', '1-3B', '3-7B', '7-15B', '15-30B', '30-100B', '100B+']:
        cat_models = param_df[param_df['size_category'] == category].nlargest(3, 'downloads')
        if len(cat_models) > 0:
            print(f"\n  {category}:")
            for _, model in cat_models.iterrows():
                print(f"    - {model['id']}")
                print(f"      Parameters: {model['params_billions']:.2f}B")
                print(f"      Downloads: {model['downloads']:,}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("""
1. DOWNLOADS:
   - 'downloads' = Recent downloads (last ~30 days)
   - 'downloadsAllTime' = Total cumulative downloads since creation
   - Use 'downloads' to track current popularity trends
   - Use 'downloadsAllTime' to understand historical impact

2. FINETUNING/MODEL TREES:
   - 'baseModels' field shows parent-child relationships
   - Structure: {'models': [...], 'relation': 'finetune'|'adapter'|'merge'}
   - Common relations: finetune, adapter, merge
   - Can trace model lineage through this field
   - Not all models have this field populated (~minority have it)

3. PARAMETER COUNTS:
   - Located in 'safetensors.total' field
   - Also get breakdown by data type in 'safetensors.parameters'
   - Stored as raw count (e.g., 8053063680 = 8.05B parameters)
   - Different data types (BF16, F16, F32, I8, etc.) indicate precision
   - Lower precision (I8, F16) often indicates quantization for efficiency
""")
