import json
import re
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from collections import defaultdict
from gpt_retrieval.call_gpt import *

OPENAI_API_KEY = ""


class SingleModalEvaluator:
    """单模态图像质量评估器"""
    
    def __init__(self, prompt_path, gen_base_path, output_dir, modal, max_workers=8, batch_size=20):
        self.prompt_path = prompt_path
        self.gen_base_path = gen_base_path
        self.output_dir = output_dir
        self.modal = modal  # 指定的单个模态：'mm', 'cot', 'dir', 'txt', 'img'
        self.max_workers = max_workers
        self.batch_size = batch_size
        
        # 线程安全的结果存储
        self.detailed_results = []
        self.summary_results = []
        self.results_lock = Lock()
        
        # 统计信息
        self.stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'start_time': None,
            'missing_images': 0
        }
        self.stats_lock = Lock()
        
        # 预加载数据
        self.prompt_data = {}
        self.grouped_questions_cache = {}
        
    def get_image_path(self, prompt_id, modal):
        """获取图像路径，支持大小写变体"""
        base_path = f"{self.gen_base_path}/{prompt_id}_{modal}.png"
        
        # 首先尝试原始路径
        if os.path.exists(base_path):
            return base_path
            
        # 尝试首字母大写的modal
        capitalized_modal = modal.capitalize()
        capitalized_path = f"{self.gen_base_path}/{prompt_id}_{capitalized_modal}.png"
        if os.path.exists(capitalized_path):
            return capitalized_path
            
        # 尝试全大写的modal
        upper_modal = modal.upper()
        upper_path = f"{self.gen_base_path}/{prompt_id}_{upper_modal}.png"
        if os.path.exists(upper_path):
            return upper_path
            
        # 如果都不存在，返回原始路径（用于错误处理）
        return base_path
        
    def load_data_once(self):
        print(f"📚 预加载数据 (模态: {self.modal})...")
        
        # 加载提示数据
        with open(self.prompt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    self.prompt_data[data.get("id")] = data
        
        # 预加载所有类别的问题数据
        categories = set(data.get("category") for data in self.prompt_data.values())
        for category in categories:
            if category:
                question_file_path = f""
                if os.path.exists(question_file_path):
                    with open(question_file_path, "r", encoding="utf-8") as f:
                        self.grouped_questions_cache[category] = json.load(f)
                        
        print(f"✅ 已加载 {len(self.prompt_data)} 个提示和 {len(self.grouped_questions_cache)} 个类别的问题数据")
    
    def check_images_exist(self):
        """检查指定模态的图像文件是否存在"""
        print(f"🔍 检查 {self.modal} 模态的图像文件...")
        missing_count = 0
        found_variants = {"original": 0, "capitalized": 0, "upper": 0}
        
        for prompt_id in self.prompt_data.keys():
            img_path = self.get_image_path(prompt_id, self.modal)
            
            # 检查实际找到的是哪种变体
            if os.path.exists(img_path):
                if f"_{self.modal}.png" in img_path:
                    found_variants["original"] += 1
                elif f"_{self.modal.capitalize()}.png" in img_path:
                    found_variants["capitalized"] += 1
                elif f"_{self.modal.upper()}.png" in img_path:
                    found_variants["upper"] += 1
            else:
                print(f"⚠️  {prompt_id}_{self.modal}.png not found (尝试了所有大小写变体)")
                missing_count += 1
        
        with self.stats_lock:
            self.stats['missing_images'] = missing_count
            
        if missing_count > 0:
            print(f"⚠️  发现 {missing_count} 个缺失的 {self.modal} 模态图像文件")
        else:
            print(f"✅ 所有 {self.modal} 模态图像文件都存在")
            
        # 显示找到的文件名格式统计
        if any(found_variants.values()):
            print(f"📊 文件名格式统计:")
            if found_variants["original"] > 0:
                print(f"   - 原始格式 ({self.modal}): {found_variants['original']} 个")
            if found_variants["capitalized"] > 0:
                print(f"   - 首字母大写 ({self.modal.capitalize()}): {found_variants['capitalized']} 个")
            if found_variants["upper"] > 0:
                print(f"   - 全大写 ({self.modal.upper()}): {found_variants['upper']} 个")
    
    def create_evaluation_tasks(self):
        """创建评估任务列表（仅针对指定模态）"""
        tasks = []
        
        for prompt_id, prompt_data in self.prompt_data.items():
            class_name = prompt_data.get("category")
            if class_name not in self.grouped_questions_cache:
                print(f"⚠️  类别 {class_name} 的问题数据未找到，跳过 {prompt_id}")
                continue
                
            grouped_questions = self.grouped_questions_cache[class_name]
            if prompt_id not in grouped_questions:
                print(f"⚠️  {prompt_id} 不在问题数据中，跳过")
                continue
            
            prompt_questions = grouped_questions[prompt_id]
            questions = prompt_questions['questions']
            
            # 使用新的路径获取方法，支持大小写变体
            generated_img_path = self.get_image_path(prompt_id, self.modal)
            if not os.path.exists(generated_img_path):
                print(f"⚠️  跳过 {prompt_id}_{self.modal}.png (文件不存在，已尝试所有大小写变体)")
                continue
            
            # 为每个问题创建任务（仅针对指定模态）
            for question_id, question_info in questions.items():
                task = {
                    'prompt_id': prompt_id,
                    'question_id': question_id,
                    'modal': self.modal,
                    'prompt_data': prompt_data,
                    'question_info': question_info,
                    'txt_reference': prompt_questions['txt_reference'],
                    'generated_img_path': generated_img_path
                }
                tasks.append(task)
        
        with self.stats_lock:
            self.stats['total_tasks'] = len(tasks)
            
        print(f"📋 创建了 {len(tasks)} 个 {self.modal} 模态评估任务")
        return tasks
    
    def evaluate_single_task(self, task):
        """评估单个任务"""
        try:
            # 提取任务信息
            prompt_id = task['prompt_id']
            question_id = task['question_id']
            modal = task['modal']
            question_info = task['question_info']
            txt_reference = task['txt_reference']
            generated_img_path = task['generated_img_path']
            
            question_content = question_info['content']
            question_type = question_info['modal']  # "T" 或 "I"
            question_ref_content = question_info.get('txt_reference', txt_reference)
            ref_img_path = question_info.get('img_path')
            question_class = question_info.get('class_id')
            
            # 准备评估客户端
            eval_client = MultimodalRetrievalClient(OPENAI_API_KEY, model="gpt-5")
            
            # 根据问题类型准备输入
            if question_type == "T":
                sys_prompt = f'You are a strict quality inspector for generated images. Based on the Reference: {question_ref_content}, evaluate whether the generated image meets the quality standards. Please respond with True or False and provide concise, key reasoning for your assessment.'
                img_paths = [generated_img_path]
                
            elif question_type == "I":
                sys_prompt = f'You are a strict quality inspector for generated images. Based on the Reference: {question_ref_content}, compare the reference image <IMAGE_0> with the generated image to evaluate quality. Please respond with True or False and provide concise, key reasoning for your assessment.'
                full_ref_img_path = f""
                img_paths = [full_ref_img_path, generated_img_path]
            else:
                return {
                    'status': 'error',
                    'task': task,
                    'error': f"Unknown question type: {question_type}"
                }
            
            # 发送评估请求
            evaluation_response = eval_client.send_single_message(
                text=question_content, 
                image_paths=img_paths, 
                system_prompt=sys_prompt
            )
            
            # 解析评估结果
            if 'True' in evaluation_response:
                score = True
            elif 'False' in evaluation_response:
                score = False
            else:
                score = None
            
            # 更新统计
            with self.stats_lock:
                self.stats['completed_tasks'] += 1
                if self.stats['completed_tasks'] % 5 == 0:
                    elapsed = time.time() - self.stats['start_time']
                    rate = self.stats['completed_tasks'] / elapsed
                    remaining = (self.stats['total_tasks'] - self.stats['completed_tasks']) / rate
                    print(f"⏳ 进度: {self.stats['completed_tasks']}/{self.stats['total_tasks']} "
                          f"({self.stats['completed_tasks']/self.stats['total_tasks']*100:.1f}%) "
                          f"预计剩余: {remaining/60:.1f}分钟")
            
            return {
                'status': 'success',
                'task': task,
                'evaluation_response': evaluation_response,
                'score': score,
                'question_class': question_class,
                'question_type': question_type,
                'question_content': question_content,
                'txt_reference': question_ref_content,
                'ref_img_path': ref_img_path
            }
            
        except Exception as e:
            with self.stats_lock:
                self.stats['failed_tasks'] += 1
            
            return {
                'status': 'error',
                'task': task,
                'error': str(e)
            }
    
    def process_results(self, results):
        """处理并组织结果"""
        print("📊 组织结果数据...")
        
        # 按prompt_id分组结果
        prompt_results = defaultdict(lambda: {
            'detailed': {'questions': {}},
            'summary': {'question_evaluations': []}
        })
        
        for result in results:
            if result['status'] != 'success':
                continue
                
            task = result['task']
            prompt_id = task['prompt_id']
            question_id = task['question_id']
            modal = task['modal']
            prompt_data = task['prompt_data']
            
            # 初始化prompt级别的信息
            if 'prompt_id' not in prompt_results[prompt_id]['detailed']:
                prompt_results[prompt_id]['detailed'].update({
                    'prompt_id': prompt_id,
                    'prompt_en': prompt_data.get('prompt_en'),
                    'txt_reference': task['txt_reference'],
                    'modal': modal  # 添加模态信息
                })
                prompt_results[prompt_id]['summary'].update({
                    'prompt_id': prompt_id,
                    'prompt_en': prompt_data.get('prompt_en'),
                    'modal': modal  # 添加模态信息
                })
            
            # 初始化question级别的信息
            if question_id not in prompt_results[prompt_id]['detailed']['questions']:
                prompt_results[prompt_id]['detailed']['questions'][question_id] = {
                    'question_content': result['question_content'],
                    'question_type': result['question_type'],
                    'question_class': result['question_class'],
                    'txt_reference': result['txt_reference'],
                    'ref_img_path': result['ref_img_path'],
                    'evaluation_response': result['evaluation_response'],
                    'score': result['score']
                }
                
                # 在summary中添加该问题
                summary_question = {
                    'question_id': question_id,
                    'question_class': result['question_class'],
                    'question_content': result['question_content'],
                    'question_type': result['question_type'],
                    'score': result['score']
                }
                prompt_results[prompt_id]['summary']['question_evaluations'].append(summary_question)
        
        # 转换为列表格式
        detailed_results = [data['detailed'] for data in prompt_results.values()]
        summary_results = [data['summary'] for data in prompt_results.values()]
        
        return detailed_results, summary_results
    
    def save_results(self, detailed_results, summary_results):
        """保存结果到文件"""
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 保存详细结果
        detailed_output_path = f"{self.output_dir}/detailed_evaluation_results_{self.modal}_modal.json"
        with open(detailed_output_path, "w", encoding="utf-8") as f:
            json.dump(detailed_results, f, ensure_ascii=False, indent=2)
        
        # 保存摘要结果
        summary_output_path = f"{self.output_dir}/summary_evaluation_results_{self.modal}_modal.json"
        with open(summary_output_path, "w", encoding="utf-8") as f:
            json.dump(summary_results, f, ensure_ascii=False, indent=2)
        
        print(f"💾 详细结果保存到: {detailed_output_path}")
        print(f"💾 摘要结果保存到: {summary_output_path}")
        
        return detailed_output_path, summary_output_path
    
    def run_single_modal_evaluation(self):
        """运行单模态评估"""
        print(f"🚀 开始 {self.modal} 模态图像质量评估")
        print("=" * 60)
        
        # 预加载数据
        self.load_data_once()
        
        # 检查图像
        self.check_images_exist()
        
        # 创建任务
        tasks = self.create_evaluation_tasks()
        if not tasks:
            print(f"❌ 没有找到可执行的 {self.modal} 模态任务")
            return None, None
        
        # 开始计时
        with self.stats_lock:
            self.stats['start_time'] = time.time()
        
        print(f"🔧 使用 {self.max_workers} 个线程并行处理")
        print(f"📦 批处理大小: {self.batch_size}")
        print()
        
        all_results = []
        
        # 分批处理任务
        for i in range(0, len(tasks), self.batch_size):
            batch_tasks = tasks[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1
            total_batches = (len(tasks) + self.batch_size - 1) // self.batch_size
            
            print(f"🔄 处理批次 {batch_num}/{total_batches} ({len(batch_tasks)} 个任务)")
            
            # 并行执行当前批次
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_task = {executor.submit(self.evaluate_single_task, task): task 
                                for task in batch_tasks}
                
                batch_results = []
                for future in as_completed(future_to_task):
                    result = future.result()
                    batch_results.append(result)
                    
                    # 实时显示错误
                    if result['status'] == 'error':
                        task = result['task']
                        print(f"❌ 任务失败: {task['prompt_id']}_{task['question_id']}_{task['modal']} - {result['error']}")
            
            all_results.extend(batch_results)
            
            # 批次完成后的统计
            success_count = sum(1 for r in batch_results if r['status'] == 'success')
            error_count = len(batch_results) - success_count
            print(f"✅ 批次 {batch_num} 完成: {success_count} 成功, {error_count} 失败")
        
        # 处理最终结果
        print("\n📊 处理最终结果...")
        detailed_results, summary_results = self.process_results(all_results)
        
        # 保存结果
        self.save_results(detailed_results, summary_results)
        
        # 最终统计
        elapsed_time = time.time() - self.stats['start_time']
        print(f"\n📈 {self.modal} 模态评估完成统计:")
        print("=" * 40)
        print(f"⏱️  总耗时: {elapsed_time/60:.2f} 分钟")
        print(f"📋 总任务数: {self.stats['total_tasks']}")
        print(f"✅ 成功任务: {self.stats['completed_tasks']}")
        print(f"❌ 失败任务: {self.stats['failed_tasks']}")
        print(f"🖼️  缺失图像: {self.stats['missing_images']}")
        print(f"📊 处理了 {len(detailed_results)} 个提示")
        print(f"⚡ 平均速度: {self.stats['completed_tasks']/(elapsed_time/60):.1f} 任务/分钟")
        
        # 计算总问题数和评估数
        total_questions = sum(len(item['questions']) for item in detailed_results)
        total_evaluations = len(all_results)
        
        print(f"❓ 总问题数: {total_questions}")
        print(f"🔬 总评估数: {total_evaluations}")
        
        return detailed_results, summary_results


def main():
    """主函数"""
    # 直接设置参数值，不使用命令行参数
    prompt_path = ""
    gen_base_path = ""
    output_dir = ""
    modal = "dir"  # 固定使用 dir 模态
    max_workers = 1000
    batch_size = 1000
    
    print(f"🎯 {modal.upper()} 模态图像质量评估工具")
    print("=" * 60)
    print(f"📁 提示文件: {prompt_path}")
    print(f"🖼️ 生成图像路径: {gen_base_path}")
    print(f"📤 输出目录: {output_dir}")
    print(f"🎭 模态类型: {modal}")
    print(f"🔧 最大线程数: {max_workers}")
    print(f"📦 批处理大小: {batch_size}")
    print()
    
    try:
        # 创建并运行评估器
        evaluator = SingleModalEvaluator(
            prompt_path=prompt_path,
            gen_base_path=gen_base_path,
            output_dir=output_dir,
            modal=modal,
            max_workers=max_workers,
            batch_size=batch_size
        )
        
        detailed_results, summary_results = evaluator.run_single_modal_evaluation()
        
        if detailed_results is not None:
            print(f"\n🎉 {modal} 模态评估成功完成！")
        else:
            print(f"\n❌ {modal} 模态评估失败")
            
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断评估")
    except Exception as e:
        print(f"\n❌ 评估过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
