#####################################
#######       上传文件         #######
#####################################
import gradio as gr
import os
import shutil
import pandas as pd
STRUCTURED_FILE_PATH = "File/Structured"
UNSTRUCTURED_FILE_PATH = "File/Unstructured"
# 刷新非结构化类目
def refresh_label():
    return os.listdir(UNSTRUCTURED_FILE_PATH)

# 刷新结构化数据表
def refresh_data_table():
    return os.listdir(STRUCTURED_FILE_PATH)

# 上传非结构化数据
def upload_unstructured_file(files,label_name):
    if files is None:
        gr.Info("请上传文件")
    elif len(label_name) == 0:
        gr.Info("请输入类目名称")
    # 判断类目是否存在
    elif label_name in os.listdir(UNSTRUCTURED_FILE_PATH):
        gr.Info(f"{label_name}类目已存在")
    else:
        try:
            if not os.path.exists(os.path.join(UNSTRUCTURED_FILE_PATH,label_name)):
                os.mkdir(os.path.join(UNSTRUCTURED_FILE_PATH,label_name))
            for file in files:
                print(file)
                file_path = file.name
                file_name = os.path.basename(file_path)
                destination_file_path = os.path.join(UNSTRUCTURED_FILE_PATH,label_name,file_name)
                shutil.move(file_path,destination_file_path)
            gr.Info(f"文件已上传至{label_name}类目中，请前往创建知识库")
        except:
            gr.Info(f"请勿重复上传")

# 上传结构化数据
def upload_structured_file(files,label_name):
    if files is None:
        gr.Info("请上传文件")
        return
    if len(label_name) == 0:
        gr.Info("请输入数据表名称")
        return
    # 判断数据表是否存在
    table_path = os.path.join(STRUCTURED_FILE_PATH, label_name)
    if os.path.exists(table_path):
        gr.Info(f"{label_name} 数据表已存在")
        return
    
    try:
        # 创建数据表文件夹
        os.makedirs(table_path, exist_ok=True)

        for file in files:
            file_path = file.name
            file_name = os.path.basename(file_path)
            destination_file_path = os.path.join(table_path, file_name)
            
            # 直接存储原始文件
            shutil.move(file_path, destination_file_path)

        gr.Info(f"文件已上传至 {label_name} 数据表中，可前往创建知识库")
    
    except Exception as e:
        gr.Info(f"文件上传失败: {str(e)}")

# 实时更新结构化数据表
def update_datatable():
    return gr.update(choices=os.listdir(STRUCTURED_FILE_PATH))


# 实时更新非结构化类目
def update_label():
    return gr.update(choices=os.listdir(UNSTRUCTURED_FILE_PATH))

# 删除类目
def delete_label(label_name):
    if label_name is not None:
        for label in label_name:
            folder_path = os.path.join(UNSTRUCTURED_FILE_PATH,label)
            if os.path.exists(folder_path):
                shutil.rmtree(folder_path)
                gr.Info(f"{label}类目已删除")

# 删除数据表
def delete_data_table(table_name):
    if table_name is not None:
        for table in table_name:
            folder_path = os.path.join(STRUCTURED_FILE_PATH,table)
            if os.path.exists(folder_path):
                shutil.rmtree(folder_path)
                gr.Info(f"{table}数据表已删除")