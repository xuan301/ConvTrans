# 项目名称

本项目用于处理时间序列数据并进行相关分析和建模。

## 环境设置

1. 创建并激活 Conda 环境：
    ```bash
    conda create -n timeser python=3.8
    conda activate timeser
    ```

2. 安装依赖包：
    ```bash
    pip install -r requirements.txt
    pip install openpyxl
    ```

## 生成数据集

1. 进入 `process_data` 目录：
    ```bash
    cd process_data
    ```

2. 新建 `outputs` 文件夹，并将各个类别的数据文件放入该文件夹：
    ```bash
    mkdir outputs
    # 将各个类别的数据文件放入 outputs 文件夹
    ```

3. 运行数据集生成脚本：
    ```bash
    python generate_dataset.py
    ```

4. 将生成的 `ProcessedData` 文件夹复制到 `Dataset/UEA/Multivariate_ts` 目录下：
    ```bash
    cp -r ProcessedData ../Dataset/UEA/Multivariate_ts
    ```

## 运行主程序

进入项目根目录，运行主程序：
```bash
python main.py
