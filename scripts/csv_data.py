import pandas as pd

df = pd.read_csv("datasets/final/final_combined_2.csv")

print("===== ATTACHMENT COVERAGE REPORT =====")

# Rows where attachment_text is non-empty
mask_has_attachment = (df["attachment_text"]
                       .fillna("")
                       .astype(str)
                       .str.strip()
                       .str.len() > 0)

attach_df = df[mask_has_attachment]

print(f"Total rows              : {len(df):,}")
print(f"Rows WITH attachments   : {len(attach_df):,}")
print(f"Attachment coverage     : {len(attach_df)/len(df)*100:.2f}%")

# Distribution of attachment sizes
attach_sizes = attach_df["attachment_text"].fillna("").astype(str).str.len()
print("\nAttachment text length stats:")
print(attach_sizes.describe())

# Detect suspicious cases where attachment exists in header but parser extracted nothing
mask_no_attachment_text = ~mask_has_attachment
suspect_rows = df[
    mask_no_attachment_text
    & df["headers_raw"].astype(str).str.contains("Content-Disposition:", case=False, na=False)
]

print("\nRows with Content-Disposition BUT no extracted attachment:")
print(len(suspect_rows))

print("\nSample suspicious rows:")
print(suspect_rows[["subject", "from_email"]].head(10))
c = df["attachment_text"].isna().sum()
print(f"Total {c} NaNs in attachment_text")