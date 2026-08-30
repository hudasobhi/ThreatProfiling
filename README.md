# Data Setup

## Dataset

ShadowDetectAI is evaluated on the **CMU CERT Insider Threat Test Dataset r6.2**.

- **Source:** Carnegie Mellon University CERT Division
- **Download URL:** https://kilthub.cmu.edu/articles/dataset/Insider_Threat_Test_Dataset/12841247
- **License:** See CMU dataset page for terms of use.

The dataset is publicly available but requires registration on the CMU KiltHub platform.
Raw data files must **not** be committed to this repository.

## Download Instructions

Run the automated download script after registering and obtaining your API token:

```bash
python data/download.py --token YOUR_KILTHUB_TOKEN --output data/raw/
```

Alternatively, download manually from the URL above and place the CSV files in `data/raw/`:

```
data/raw/
├── logon.csv
├── device.csv
├── email.csv
├── file.csv
├── http.csv
└── psychometric.csv
```

## Preprocessing

After downloading, run Algorithm 1 (feature extraction and graph construction):

```bash
python data/preprocess.py --config config/data_config.yaml
```

This produces processed partition files in `data/processed/`:

```
data/processed/
├── train_windows.pt      # 96,000 user-windows, 42 malicious
├── val_windows.pt        # 20,600 user-windows, 9 malicious
├── test_windows.pt       # 20,600 user-windows, 9 malicious
├── loio_folds/           # 5 LOIO folds (Table 10)
│   ├── fold_1.pt
│   ├── fold_2.pt
│   ├── fold_3.pt
│   ├── fold_4.pt
│   └── fold_5.pt
└── graph_metadata.json   # Node set and adjacency metadata
```

## Partition Design

Partitions are temporally contiguous with no gap month (acknowledged limitation in manuscript Section 6.1):

| Partition | Months  | Windows | Malicious Windows | Malicious Identities |
|-----------|---------|---------|-------------------|----------------------|
| Training  | 1–13    | 96,000  | 42                | 3                    |
| Validation| 14–15   | 20,600  | 9                 | 1                    |
| Test      | 16–18   | 20,600  | 9                 | 1                    |

TGN memory states are reset to zero at each partition boundary (cold-start).
No identity spans more than one partition.

## Behavioral Proxy Features

The CERT r6.2 dataset does not include raw keystroke or mouse telemetry.
Keystroke and mouse channels are realized as **log-derived proxy features**:

- **Keystroke proxy:** inter-event timing intervals from logon-session activity records
  approximate dwell time (Eq. 11) and flight time (Eq. 12).
- **Mouse proxy:** device-event frequency distributions and inter-event interval statistics
  approximate mouse movement entropy (Eq. 14).
- **Application usage:** file, email, and HTTP activity logs (direct, not proxy).

Readers should not interpret reported metrics as validation against genuine biometric sensor data.
