import pandas as pd

metadata = pd.read_csv("data/seed/seed_metadata_v003_3h.csv")

search_columns = [
    "content_type_of_movement",
    "content_body_position",
    "filename",
    "move_name",
]

text = (
    metadata[search_columns]
    .fillna("")
    .astype(str)
    .agg(" ".join, axis=1)
)

crawl_mask = text.str.contains(
    r"crawling|crawl|on all fours|hands and knees",
    case=False,
    regex=True,
)

crawl = (
    metadata.loc[crawl_mask]
    .drop_duplicates(subset="move_g1_path")
)

crawl.to_csv(
    "data/seed/seed_crawl_metadata.csv",
    index=False,
)

print("Crawl clips:", len(crawl))