from flask import Flask, render_template, request, send_file, jsonify, session, current_app, url_for
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import os
import numpy as np
import json
import warnings

from SpammerGroupDetection.data_loading import load_data
import warnings
from SpammerGroupDetection.evaluation import evaluate_all_restaurants, set_global_confidence_threshold_and_identify_spam_groups
from SpammerGroupDetection.visualization import visualize_restaurant_and_reviewers, visualize_spam_group_global_relationships
from SpammerGroupDetection.group_cal import print_spam_groups_info
from Detector.SpamDetect import text_input,prepare_pipe,spam_detect

app = Flask(__name__)
app.secret_key = '123'

save_dir = "/plots/"

global_pipe = None
warnings.filterwarnings('ignore')

# 在upload_csv路由中保存identified_spam_groups到本地文件
def save_identified_spam_groups(identified_spam_groups):
    with open('identified_spam_groups.json', 'w') as f:
        json.dump(identified_spam_groups, f)

# 在generate_image路由中加载identified_spam_groups
def load_identified_spam_groups():
    with open('identified_spam_groups.json', 'r') as f:
        return json.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/prepare_pipeline', methods=['GET'])
def prepare_pipeline():
    # 这里准备管道，例如使用你的 prepare_pipe() 函数
    global global_pipe
    global_pipe = prepare_pipe()
    return jsonify({'message': 'Pipeline prepared successfully'})

@app.route('/your_backend_endpoint', methods=['POST'])
def process_message():
    global global_pipe

    # 获取从前端发送过来的 JSON 数据
    data = request.json
    
    # 从 JSON 数据中获取用户输入的消息
    text = data.get('message')
    
    # 检查管道是否已经准备好
    if global_pipe is None:
        return jsonify({'error': 'Pipeline not prepared. Please prepare the pipeline first.'}), 400
    
    # 使用保存的管道进行文本处理
    formatted_text = text_input(text)
    output = spam_detect(pipe=global_pipe, text=formatted_text)
    
    # 根据输出的标签确定回复内容
    if output['label'] == 'LABEL_1':
        response = f"This is a spam. The confidence score is {output['score']:.2f}"
    elif output['label'] == 'LABEL_0':
        response = f"This is not a spam. The confidence score is {output['score']:.2f}"
    else:
        response = "Unable to determine spam status."
    
    # 返回处理结果
    return jsonify({'response': response})

@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    # 检查是否有上传的文件
    if 'csv_file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['csv_file']
    
    # 检查文件名是否为空
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    # 保存上传的文件到服务器上
    file_path = os.path.join(app.root_path, file.filename)
    file.save(file_path)
    data = load_data(file_path)
    session['file_path'] = file_path
    ks = range(2, 15)
    all_results = evaluate_all_restaurants(data, ks)
    # all_results_size = sum(len(v) for v in all_results.values())
    # print("Size of all_results:", all_results_size)

    identified_spam_groups = set_global_confidence_threshold_and_identify_spam_groups(all_results, percentile=90)
    # print("Size of identified_spam_groups before:", len(identified_spam_groups))
    save_identified_spam_groups(identified_spam_groups)
    # print("Size of identified_spam_groups mid:", len(identified_spam_groups))
    fresult = print_spam_groups_info(identified_spam_groups)
    # print("Size of identified_spam_groups after:", len(identified_spam_groups))
    

    # print("Identified spam groups:")
    # for group in identified_spam_groups:
    #     print(group)

    # print("Type of fresult:", type(fresult))
    # print("len of fresult:", len(fresult))
    
    
    # 直接将结果转换为 JSON 字符串，然后返回
    json_results = json.dumps(fresult, default=str)
    print("Session data in upload_csv:", session)

    return jsonify(results=json_results)

@app.route('/generate_image', methods=['POST'])
def generate_image():
    # 接收前端发送的用户输入内容

    identified_spam_groups = load_identified_spam_groups()
    file_path = session.get('file_path')
    data = load_data(file_path)
    user_input_str = request.json.get('user_input')  # 获取用户输入的字符串
    try:
        user_input = int(user_input_str)  # 尝试将字符串转换为整数
    except ValueError:
        # 如果无法转换为整数，可以返回一个错误或者设置默认值
        return jsonify({'error': 'Invalid input. Please enter a valid integer.'}), 400

    visualize_restaurant_and_reviewers(app, data, identified_spam_groups, user_input)
    visualize_spam_group_global_relationships(app, data, identified_spam_groups, user_input)

    # 生成图片文件的路径
    restaurant_image_path = url_for('static', filename=f"plots/restaurant_{user_input}_reviewers.png")
    spam_image_path = url_for('static', filename=f"plots/spam_group_relationships_restaurant_{user_input}.png")


    return jsonify({'restaurant_image_path': restaurant_image_path, 'spam_image_path': spam_image_path})


if __name__ == '__main__':
    app.run(debug=False)