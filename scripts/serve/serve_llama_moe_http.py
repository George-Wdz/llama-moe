#!/usr/bin/env python
"""Minimal HTTP server for local LLaMA-MoE inference.

This intentionally uses only Python stdlib plus torch/transformers so it can
run in the existing smoe environment without installing FastAPI/uvicorn.
"""

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_DIR = "/home/wdz/BT/MoE/models/LLaMA-MoE-v1-3_5B-2_8"


class ModelRunner:
    def __init__(self, model_dir, device, dtype):
        self.model_dir = model_dir
        self.device = device
        self.dtype = dtype
        self.lock = threading.Lock()

        torch_dtype = {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[dtype]

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        self.model.eval()
        self.model.to(device)

    @torch.inference_mode()
    def generate(
        self,
        prompt,
        max_new_tokens=80,
        temperature=0.0,
        top_p=1.0,
        repetition_penalty=1.0,
    ):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        do_sample = temperature is not None and temperature > 0

        with self.lock:
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        input_len = inputs["input_ids"].shape[-1]
        generated_ids = output_ids[0, input_len:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        full_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return {
            "prompt": prompt,
            "generated_text": generated_text,
            "full_text": full_text,
            "input_tokens": int(input_len),
            "output_tokens": int(generated_ids.shape[-1]),
        }


def make_handler(runner):
    class Handler(BaseHTTPRequestHandler):
        server_version = "LlamaMoEHTTP/0.1"

        def _send_json(self, status, payload):
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()
            self.wfile.write(data)

        def do_OPTIONS(self):
            self._send_json(200, {"ok": True})

        def do_GET(self):
            if self.path == "/health":
                self._send_json(
                    200,
                    {
                        "status": "ok",
                        "model_dir": runner.model_dir,
                        "device": runner.device,
                        "dtype": runner.dtype,
                    },
                )
                return
            self._send_json(
                404,
                {
                    "error": "not_found",
                    "message": "Use GET /health or POST /generate",
                },
            )

        def do_POST(self):
            if self.path != "/generate":
                self._send_json(404, {"error": "not_found"})
                return

            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(content_length).decode("utf-8")
                request = json.loads(body) if body else {}
                prompt = request["prompt"]
                result = runner.generate(
                    prompt=prompt,
                    max_new_tokens=int(request.get("max_new_tokens", 80)),
                    temperature=float(request.get("temperature", 0.0)),
                    top_p=float(request.get("top_p", 1.0)),
                    repetition_penalty=float(request.get("repetition_penalty", 1.0)),
                )
                self._send_json(200, result)
            except KeyError:
                self._send_json(400, {"error": "bad_request", "message": "Missing prompt"})
            except Exception as exc:
                self._send_json(
                    500,
                    {
                        "error": type(exc).__name__,
                        "message": str(exc),
                    },
                )

        def log_message(self, fmt, *args):
            print("%s - %s" % (self.address_string(), fmt % args), flush=True)

    return Handler


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8008)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="bfloat16",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    runner = ModelRunner(args.model_dir, args.device, args.dtype)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(runner))
    print(f"Serving LLaMA-MoE on http://{args.host}:{args.port}", flush=True)
    print("Health:   GET  /health", flush=True)
    print("Generate: POST /generate", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
