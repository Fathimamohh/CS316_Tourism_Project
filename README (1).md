# Data

`DDS_MT_Master_Sheet_2021_2025.xlsx` is the original, unmodified workbook
*Tourism Demand Indicators in the Kingdom of Saudi Arabia* (2021–2025, last update 12-04-2026),
published by the Data and Decision Support Deputy, Ministry of Tourism, as open data:
<https://mt.gov.sa/tic/open-data>.

We use two sheets:

| Sheet | Content |
|---|---|
| `D5` | Domestic tourists number & spending by destination – main cities/governorates, monthly |
| `I6` | Inbound tourists number & spending by destination – main cities/governorates, monthly |

If the file is missing, download it from the link above and save it in this folder with the
exact name `DDS_MT_Master_Sheet_2021_2025.xlsx`.

`processed/` is created by notebook 01 and contains the tidy CSV versions
(`tourism_destination_month.csv`, `tourism_national_totals.csv`).
