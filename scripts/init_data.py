import json
import os
import time

import requests

# ================= 配置区域 =================
# JSON 数据文件路径 (请确保文件存在于 scripts 目录下)
JSON_FILE_PATH = "1218_json.json"

# 后端 API 接口地址
API_URL = "http://localhost:8077/api/entities/Product/ingest"

# 批处理大小 (每批发送多少条数据)
# 建议设置在 50-100 之间，避免单次请求过大导致 HTTP 超时或数据库压力过大
BATCH_SIZE = 10


# ===========================================

def init_data():
    """
    读取本地 JSON 文件并将数据批量推送至后端服务
    """
    # 1. 检查文件是否存在
    if not os.path.exists(JSON_FILE_PATH):
        print(f"❌ [错误] 找不到文件: {JSON_FILE_PATH}")
        print("   请确认 scripts/1218.json 文件是否存在于项目目录中。")
        return

    # 2. 读取并解析 JSON 文件
    print(f"📂 正在读取文件: {JSON_FILE_PATH} ...")
    try:
        with open(JSON_FILE_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 兼容处理: 无论 JSON 是 {"products": [...]} 格式还是直接的 [...] 列表格式
        if isinstance(data, dict) and "products" in data:
            product_list = data["products"]
        elif isinstance(data, list):
            product_list = data
        else:
            print("❌ [错误] JSON 结构不符合预期，未找到列表数据。")
            return

        if not product_list:
            print("⚠️ [警告] 数据列表为空，无需执行导入。")
            return

    except json.JSONDecodeError as e:
        print(f"❌ [错误] JSON 解析失败: {e}")
        return
    except Exception as e:
        print(f"❌ [错误] 读取文件失败: {e}")
        return

    # 3. 打印统计信息
    total_count = len(product_list)
    total_batches = (total_count + BATCH_SIZE - 1) // BATCH_SIZE

    print("-" * 60)
    print(f"📊 数据统计结果:")
    print(f"   - 总条数: {total_count}")
    print(f"   - 批次大小: {BATCH_SIZE}")
    print(f"   - 预计批次: {total_batches}")
    print("-" * 60)

    # 4. 开始分批发送
    headers = {"Content-Type": "application/json"}
    success_total = 0
    fail_total = 0
    start_time = time.time()

    print("🚀 开始执行批量导入任务...\n")

    for i in range(0, total_count, BATCH_SIZE):
        # 切片获取当前批次数据
        batch_data = product_list[i: i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        current_range_str = f"{i + 1}~{min(i + BATCH_SIZE, total_count)}"

        print(f"📦 [批次 {batch_num}/{total_batches}] 发送数据 ({current_range_str}) ... ", end="", flush=True)

        try:
            # 发送 POST 请求
            response = requests.post(
                API_URL,
                json={
                    "items": batch_data,
                    "group_id": "default",
                    "concurrency": 5,
                    "auto_link": False,
                    "score_threshold": 0.3,
                },
                headers=headers,
            )

            # 5. 处理并打印响应
            if response.status_code == 200:
                res_json = response.json()
                # 简单打印成功信息，如果需要详细日志可改为 print(res_json)
                print(f"✅ 成功")
                print(f"    └─ 服务端响应: {res_json}")
                success_total += len(batch_data)
            else:
                print(f"❌ 失败 (Status: {response.status_code})")
                print(f"    └─ 错误详情: {response.text}")
                fail_total += len(batch_data)

        except requests.exceptions.ConnectionError:
            print(f"❌ 连接失败")
            print(f"    └─ 无法连接到 {API_URL}，请检查后端服务是否已启动。")
            fail_total += len(batch_data)
        except Exception as e:
            print(f"❌ 异常发生")
            print(f"    └─ {str(e)}")
            fail_total += len(batch_data)

        # 简单的限流，防止请求过于密集
        time.sleep(0.5)

    # 6. 任务总结
    duration = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"🏁 任务执行完毕")
    print(f"⏱️  耗时: {duration:.2f} 秒")
    print(f"🟢 成功导入: {success_total} 条")
    if fail_total > 0:
        print(f"🔴 导入失败: {fail_total} 条")
    else:
        print(f"✨ 全部成功!")
    print("=" * 60)


if __name__ == "__main__":
    init_data()
