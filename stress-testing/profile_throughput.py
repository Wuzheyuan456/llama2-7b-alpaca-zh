#!/usr/bin/env python3
# profile_throughput.py
import argparse, csv, json, time, sys, statistics
from typing import List, Tuple
import requests
from transformers import AutoTokenizer
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

# ---------- 参数解析 ----------
def parse_args():
    parser = argparse.ArgumentParser(description="OpenAI-Compatible API 压测")
    parser.add_argument("--backend", type=str, default="openai-chat", help="兼容后端，固定 openai-chat")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8800)
    parser.add_argument("--tokenizer", type=str, required=True, help="tokenizer 路径")
    parser.add_argument("--num-prompts", type=int, default=1000, help="总请求数")
    parser.add_argument("--prompt-tokens", type=int, default=512, help="prompt 长度")
    parser.add_argument("--output-tokens", type=int, default=128, help="期望输出长度")
    parser.add_argument("--concurrency", type=int, default=10, help="并发线程数")
    parser.add_argument("--csv", type=str, default="result.csv", help="输出 csv")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=1.0)
    return parser.parse_args()

# ---------- 生成定长 prompt ----------

def gen_fixed_prompt( tokenizer,token_num: int) -> str:
    # 用 'hi' 重复拼到指定 token 数（简单且通用）
    hi_ids = tokenizer('hi', add_special_tokens=False)['input_ids']
    repeat = (token_num // len(hi_ids)) + 1
    prompt_ids = (hi_ids * repeat)[:token_num]
    return tokenizer.decode(prompt_ids, skip_special_tokens=True)

# ---------- 单次请求 ----------
def single_request(args,tokenizer, prompt: str, idx: int) -> Tuple[int, float, float, int]:
    url = f"http://{args.host}:{args.port}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": r"gpt-3.5-turbo",  # 与注册名一致即可
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.output_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "stream": False
    }

    st = time.perf_counter()
    try:
        resp = requests.post(url, headers=headers, json=data, timeout=120)
        resp.raise_for_status()
    except Exception as e:
        print(f"[warn] request {idx} failed: {e}")
        return 0, 0, 0, 0

    latency = time.perf_counter() - st
    content = resp.json()
    if "choices" not in content or len(content["choices"]) == 0:
        return 0, 0, 0, 0

    # 统计返回 token 数
    text = content["choices"][0]["message"]["content"]
    tok_num = len(tokenizer.encode(text, add_special_tokens=False)) + 1  # +1 for </s>
    # 首 token 时间：choices[0]["delta"]["content"] 首字符出现时间
    # 这里用总时间近似（保守），若服务端支持 stream 可更准
    ttft = latency  # 保守近似
    return tok_num, latency, ttft, 1

# ---------- 主压测逻辑 ----------
def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    prompt = gen_fixed_prompt(tokenizer, args.prompt_tokens)
    print(f"prompt={prompt[:30]}...({args.prompt_tokens} tokens)")

    tok_list: List[int] = []
    lat_list: List[float] = []
    ttft_list: List[float] = []
    ok_cnt = 0

    start_t = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(single_request, args,tokenizer, prompt, i) for i in range(args.num_prompts)]
        for fut in as_completed(futures):
            tok, lat, ttft, ok = fut.result()
            if ok:
                tok_list.append(tok)
                lat_list.append(lat)
                ttft_list.append(ttft)
                ok_cnt += 1
    total_t = time.perf_counter() - start_t

    # ---------- 统计 ----------
    total_tokens = sum(tok_list)
    throughput = total_tokens / total_t
    avg_ttft = statistics.mean(ttft_list) if ttft_list else 0
    p95_ttft = np.percentile(ttft_list, 95) if ttft_list else 0
    avg_lat = statistics.mean(lat_list) if lat_list else 0

    print("\n========== 结果 ==========")
    print(f"成功请求: {ok_cnt}/{args.num_prompts}")
    print(f"总耗时: {total_t:.2f} s")
    print(f"总生成 tokens: {total_tokens}")
    print(f"Throughput: {throughput:.2f} tokens/s")
    print(f"TTFT 平均: {avg_ttft*1000:.2f} ms")
    print(f"TTFT 95th: {p95_ttft*1000:.2f} ms")
    print(f"Request 平均延迟: {avg_lat*1000:.2f} ms")

    # ---------- 写 CSV ----------
    with open(args.csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["throughput(tok/s)", "ttft_avg(ms)", "ttft_p95(ms)", "success"])
        writer.writerow([f"{throughput:.2f}", f"{avg_ttft*1000:.2f}", f"{p95_ttft*1000:.2f}", f"{ok_cnt}/{args.num_prompts}"])
    print(f"\nCSV 已写入: {args.csv}")

if __name__ == "__main__":
    main()