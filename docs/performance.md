# LiteOCR All-Models Component Benchmark

Environment: Intel RaptorLake-S (4 threads), NVIDIA GeForce RTX 4060 Laptop GPU (device 1), synthetic random BGR images.\n\n**Note:** `PP-OCRv5_server_det` CPU FP32 at 960×960 exceeded the 5-minute timeout and is marked as error. Server models are very heavy on CPU; use GPU for those.

## PP-OCRv5_mobile_det

| Config | 320x320 | 640x640 | 960x960 |
|---|---|---|---|
| CPU FP32 | 27.23 ms | 121.77 ms | 423.33 ms |
| CPU BF16 | 36.48 ms | 155.71 ms | 610.10 ms |
| GPU1 FP32 | 3.74 ms | 6.10 ms | 11.65 ms |
| GPU1 FP16 | 2.37 ms | 5.74 ms | 11.00 ms |
| GPU1 BF16 | 12.08 ms | 20.23 ms | 22.78 ms |

## PP-OCRv5_mobile_det_int8

| Config | 320x320 | 640x640 | 960x960 |
|---|---|---|---|
| CPU INT8 | 12.21 ms | 56.24 ms | 179.16 ms |
| GPU1 INT8 | 13.92 ms | 68.89 ms | 146.06 ms |

## PP-OCRv5_server_det

| Config | 320x320 | 640x640 | 960x960 |
|---|---|---|---|
| CPU FP32 | error | error | error |
| CPU BF16 | 1376.37 ms | 4389.74 ms | 10138.90 ms |
| GPU1 FP32 | 15.02 ms | 40.91 ms | 83.71 ms |
| GPU1 FP16 | 15.16 ms | 39.87 ms | 83.02 ms |
| GPU1 BF16 | 49.01 ms | 124.08 ms | 216.64 ms |

## PP-OCRv6_medium_det

| Config | 320x320 | 640x640 | 960x960 |
|---|---|---|---|
| CPU FP32 | 214.15 ms | 1014.55 ms | 2357.21 ms |
| CPU BF16 | 258.04 ms | 1102.50 ms | 2553.15 ms |
| GPU1 FP32 | 7.93 ms | 21.45 ms | 44.55 ms |
| GPU1 FP16 | 7.74 ms | 21.49 ms | 44.62 ms |
| GPU1 BF16 | 19.47 ms | 56.87 ms | 119.20 ms |

## PP-OCRv6_medium_det_int8

| Config | 320x320 | 640x640 | 960x960 |
|---|---|---|---|
| CPU INT8 | 100.36 ms | 456.78 ms | 1073.49 ms |
| GPU1 INT8 | 101.01 ms | 452.40 ms | 1069.51 ms |

## PP-OCRv6_small_det

| Config | 320x320 | 640x640 | 960x960 |
|---|---|---|---|
| CPU FP32 | 29.98 ms | 141.40 ms | 354.15 ms |
| CPU BF16 | 39.58 ms | 167.21 ms | 390.73 ms |
| GPU1 FP32 | 3.93 ms | 9.92 ms | 18.47 ms |
| GPU1 FP16 | 3.86 ms | 9.65 ms | 18.76 ms |
| GPU1 BF16 | 5.70 ms | 15.37 ms | 29.49 ms |

## PP-OCRv6_small_det_int8

| Config | 320x320 | 640x640 | 960x960 |
|---|---|---|---|
| CPU INT8 | 21.16 ms | 101.87 ms | 244.13 ms |
| GPU1 INT8 | 21.49 ms | 104.61 ms | 247.32 ms |

## PP-OCRv6_tiny_det

| Config | 320x320 | 640x640 | 960x960 |
|---|---|---|---|
| CPU FP32 | 11.66 ms | 57.35 ms | 132.35 ms |
| CPU BF16 | 17.44 ms | 75.68 ms | 172.34 ms |
| GPU1 FP32 | 3.21 ms | 7.39 ms | 14.27 ms |
| GPU1 FP16 | 3.13 ms | 7.45 ms | 14.28 ms |
| GPU1 BF16 | 4.18 ms | 9.96 ms | 18.83 ms |

## PP-OCRv6_tiny_det_int8

| Config | 320x320 | 640x640 | 960x960 |
|---|---|---|---|
| CPU INT8 | 10.15 ms | 50.94 ms | 121.64 ms |
| GPU1 INT8 | 10.20 ms | 50.99 ms | 125.95 ms |

## PP-OCRv5_mobile_rec

| Config | 128x48 | 256x48 | 512x48 |
|---|---|---|---|
| CPU FP32 | 7.40 ms | 14.98 ms | 30.19 ms |
| CPU BF16 | 10.32 ms | 21.52 ms | 43.17 ms |
| GPU1 FP32 | 8.17 ms | 9.76 ms | 11.51 ms |
| GPU1 FP16 | 8.04 ms | 9.94 ms | 13.77 ms |
| GPU1 BF16 | 18.07 ms | 11.77 ms | 15.38 ms |

## PP-OCRv5_mobile_rec_int8

| Config | 128x48 | 256x48 | 512x48 |
|---|---|---|---|
| CPU INT8 | 5.50 ms | 11.07 ms | 25.83 ms |
| GPU1 INT8 | 7.11 ms | 11.88 ms | 22.10 ms |

## PP-OCRv5_server_rec

| Config | 128x48 | 256x48 | 512x48 |
|---|---|---|---|
| CPU FP32 | 713.93 ms | 1415.76 ms | 2847.45 ms |
| CPU BF16 | 869.33 ms | 1739.08 ms | 3462.61 ms |
| GPU1 FP32 | 5.45 ms | 6.30 ms | 8.59 ms |
| GPU1 FP16 | 5.11 ms | 6.37 ms | 8.13 ms |
| GPU1 BF16 | 7.42 ms | 9.31 ms | 12.60 ms |

## PP-OCRv6_medium_rec

| Config | 128x48 | 256x48 | 512x48 |
|---|---|---|---|
| CPU FP32 | 38.53 ms | 80.97 ms | 161.10 ms |
| CPU BF16 | 53.84 ms | 108.70 ms | 218.77 ms |
| GPU1 FP32 | 4.39 ms | 5.82 ms | 8.52 ms |
| GPU1 FP16 | 4.43 ms | 5.63 ms | 8.10 ms |
| GPU1 BF16 | 13.99 ms | 12.61 ms | 12.44 ms |

## PP-OCRv6_medium_rec_int8

| Config | 128x48 | 256x48 | 512x48 |
|---|---|---|---|
| CPU INT8 | 16.72 ms | 32.59 ms | 68.29 ms |
| GPU1 INT8 | 16.65 ms | 32.83 ms | 67.72 ms |

## PP-OCRv6_small_rec

| Config | 128x48 | 256x48 | 512x48 |
|---|---|---|---|
| CPU FP32 | 9.58 ms | 17.98 ms | 35.53 ms |
| CPU BF16 | 12.23 ms | 24.59 ms | 47.91 ms |
| GPU1 FP32 | 4.10 ms | 5.31 ms | 7.28 ms |
| GPU1 FP16 | 4.09 ms | 5.15 ms | 7.77 ms |
| GPU1 BF16 | 4.55 ms | 5.67 ms | 8.12 ms |

## PP-OCRv6_small_rec_int8

| Config | 128x48 | 256x48 | 512x48 |
|---|---|---|---|
| CPU INT8 | 5.38 ms | 10.02 ms | 21.19 ms |
| GPU1 INT8 | 5.19 ms | 10.24 ms | 20.38 ms |

## PP-OCRv6_tiny_rec

| Config | 128x48 | 256x48 | 512x48 |
|---|---|---|---|
| CPU FP32 | 1.69 ms | 3.28 ms | 7.23 ms |
| CPU BF16 | 2.38 ms | 5.43 ms | 9.97 ms |
| GPU1 FP32 | 1.43 ms | 1.58 ms | 2.81 ms |
| GPU1 FP16 | 1.73 ms | 1.57 ms | 2.75 ms |
| GPU1 BF16 | 1.56 ms | 1.75 ms | 2.98 ms |

## PP-OCRv6_tiny_rec_int8

| Config | 128x48 | 256x48 | 512x48 |
|---|---|---|---|
| CPU INT8 | 1.32 ms | 2.42 ms | 5.38 ms |
| GPU1 INT8 | 1.03 ms | 2.20 ms | 5.25 ms |

