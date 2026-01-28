#!/usr/bin/env python3
import argparse
import json
import os
import time

import pandas as pd
import pyshark
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.classifiers.base_classifier import BaseClassifier
from core.classifiers.lightgbm_classifier import LightGBMClassifier
from core.classifiers.loader import Loader
from core.utils import NetworkUtils, OsUtils, PrintUtils, ThrowingArgparse


def get_self_dir():
    """Return the directory containing this script."""
    return os.path.dirname(os.path.abspath(__file__))


def parse_arguments():
    """Parse script input arguments."""
    PrintUtils.start_stage('Parsing command-line arguments')
    parser = ThrowingArgparse()
    parser.add_argument('--prompt', type=str, help='Single prompt to run inference on.')
    parser.add_argument(
        '--data',
        type=str,
        help='Path to a text file (one prompt per line) to run inference on.'
    )
    parser.add_argument(
        '--models',
        nargs='+',
        help='Paths to trained model checkpoint files (.pth). Supports multiple models.',
        default=['lstm_binary_classifier.pth']
    )
    parser.add_argument(
        '--norm_params',
        nargs='+',
        help='Paths to normalization parameter files (.json). Count must match models.',
        default=['lstm_binary_classifier_norm_params.json']
    )
    parser.add_argument(
        '--capture',
        type=str,
        help='Existing capture (.pcap) to reuse instead of taking a new capture.'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/inference_results.json',
        help='Output path for inference results.'
    )
    parser.add_argument(
        '-t',
        '--tlsport',
        type=int,
        default=443,
        help='Remote TLS port to capture traffic from.'
    )
    args = parser.parse_args()

    assert args.prompt or args.data, 'Either --prompt or --data must be specified.'
    assert len(args.models) == len(args.norm_params), 'Models and normalization files count mismatch.'

    PrintUtils.end_stage()
    return args


def load_input_prompts(args):
    """Load prompts either from --prompt or --data."""
    PrintUtils.start_stage('Loading input prompts')
    if args.prompt:
        prompts = [args.prompt]
    else:
        with open(args.data, 'r', encoding='utf-8') as fp:
            prompts = [line.strip() for line in fp if line.strip()]
    assert prompts, 'No prompts loaded.'
    PrintUtils.print_extra(f'Loaded {len(prompts)} prompts for inference.')
    PrintUtils.end_stage()
    return prompts


def perform_capture(args):
    """Perform a network capture or reuse an existing pcap file."""
    PrintUtils.start_stage('Network capture setup')
    captures_path = os.path.join(get_self_dir(), 'captures')
    assert OsUtils.mkdir(captures_path), 'Failed to ensure captures directory exists.'
    pcap_file = args.capture or os.path.join(captures_path, 'inference_capture.pcap')

    if args.capture:
        PrintUtils.print_extra(f'Using existing capture file: {pcap_file}')
    else:
        NetworkUtils.start_sniffing_tls(pcap_file, args.tlsport)
        PrintUtils.print_extra('Capture started. Trigger the interaction now...')
        input('Press Enter once the interaction is complete.')
        NetworkUtils.stop_sniffing_tls()
        PrintUtils.print_extra(f'Capture completed and saved to {pcap_file}.')
    PrintUtils.end_stage()
    return pcap_file


def _infer_local_port_from_pcap(pcap_file, remote_port):
    """Inspect the capture to infer the client source port."""
    capture = None
    try:
        capture = pyshark.FileCapture(
            pcap_file,
            display_filter=f'tcp.port == {remote_port}'
        )
        for packet in capture:
            if not hasattr(packet, 'tls'):
                continue
            if (
                hasattr(packet.tls, 'handshake_type')
                and packet.tls.handshake_type == '1'
                and hasattr(packet, 'tcp')
            ):
                return int(packet.tcp.srcport)
    finally:
        if capture is not None:
            capture.close()
    return None


def _build_sequence_from_capture(pcap_file, prompt, local_port, remote_port):
    """Parse the capture and build a sequence dictionary."""
    cap = None
    sequence = {
        'timestamp': time.time(),
        'local_port': local_port,
        'remote_port': remote_port,
        'prompt': prompt,
        'pertubated_prompt': prompt,
        'response': '',
        'response_tokens': [],
        'response_token_count': 0,
        'response_token_count_nonempty': 0,
        'response_token_count_empty': 0,
        'temperature': 0.0,
        'data_lengths': [],
        'time_diffs': []
    }

    display_filter = f'tcp.port == {local_port} || tcp.port == {remote_port}'
    try:
        cap = pyshark.FileCapture(pcap_file, display_filter=display_filter)
        client_hello_found = False
        prev_sniff_time = None

        for packet in cap:
            if not hasattr(packet, 'tls'):
                continue

            if (
                hasattr(packet.tls, 'handshake_type')
                and packet.tls.handshake_type == '1'
                and int(packet.tcp.dstport) == remote_port
                and int(packet.tcp.srcport) == local_port
            ):
                client_hello_found = True
                prev_sniff_time = float(packet.sniff_time.timestamp())
                continue

            if not client_hello_found:
                continue

            if (
                hasattr(packet.tls, 'app_data')
                and int(packet.tcp.dstport) == local_port
                and int(packet.tcp.srcport) == remote_port
            ):
                timestamp = float(packet.sniff_time.timestamp())
                sequence['data_lengths'].append(int(packet.length))
                sequence['time_diffs'].append(timestamp - prev_sniff_time)
                prev_sniff_time = timestamp

        if not sequence['data_lengths']:
            raise Exception(f'No TLS application data found in capture {pcap_file}')

    finally:
        if cap is not None:
            cap.close()

    return sequence


def generate_sequence(pcap_file, prompt, remote_port, last_known_port=0):
    """Generate a sequence dictionary from the capture and return (sequence, local_port)."""
    PrintUtils.start_stage('Generating sequence from capture')
    local_port = _infer_local_port_from_pcap(pcap_file, remote_port)
    if local_port is None or local_port <= 0:
        local_port = last_known_port

    if local_port is None or local_port <= 0:
        raise Exception('Unable to determine the local TLS port for the capture.')

    sequence = _build_sequence_from_capture(pcap_file, prompt, local_port, remote_port)
    PrintUtils.print_extra(f'Parsed capture using local port {local_port}.')
    PrintUtils.end_stage()
    return sequence, local_port


def load_normalization(norm_path):
    """Load normalization parameters stored alongside the trained model."""
    with open(norm_path, 'r', encoding='utf-8') as fp:
        data = json.load(fp)
    if 'normalization_params' not in data or 'class_name' not in data:
        raise Exception(f'Invalid normalization file: {norm_path}')
    return data['normalization_params'], data['class_name'], data.get('args', {})


def prepare_data_loader(sequence, norm_params, batch_size=1):
    """Normalize the sequence and wrap it in Loader/DataLoader objects."""
    PrintUtils.start_stage('Preparing data for model')
    df = pd.DataFrame([{
        'prompt': sequence['prompt'],
        'time_diffs': sequence['time_diffs'],
        'data_lengths': sequence['data_lengths'],
        'target': 0
    }])

    dataset = Loader(df)
    dataset.apply_normalization(norm_params)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    PrintUtils.end_stage()
    return dataset, dataloader


def inference_model(model_path, norm_path, dataset, dataloader, device):
    """Run inference using a single model and normalization file."""
    _, class_name, _ = load_normalization(norm_path)

    if class_name == 'LightGBMClassifier':
        model = LightGBMClassifier.load(model_path)
        features = dataset.df
        probability = model.predict_proba(features)[0]
        return int(probability > 0.5), float(probability)

    model = BaseClassifier.load(model_path, device)
    model.eval()

    with torch.no_grad():
        for batch, _ in dataloader:
            batch = batch.to(device)
            logits = model(batch)
            probability = torch.sigmoid(logits).item()
            return int(probability > 0.5), float(probability)

    raise Exception('No samples available for model inference.')


def main():
    try:
        PrintUtils.print_logo()

        args = parse_arguments()
        prompts = load_input_prompts(args)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        PrintUtils.print_extra(f'Using device: {device}')

        pcap_file = perform_capture(args)
        last_local_port = 0
        results = []

        for prompt in tqdm(prompts, desc='Running prompts'):
            sequence, last_local_port = generate_sequence(
                pcap_file,
                prompt,
                args.tlsport,
                last_local_port
            )

            prompt_results = {'prompt': prompt, 'models': []}

            for model_path, norm_path in zip(args.models, args.norm_params):
                norm_params, _, _ = load_normalization(norm_path)
                dataset, dataloader = prepare_data_loader(sequence, norm_params)
                prediction, score = inference_model(model_path, norm_path, dataset, dataloader, device)
                PrintUtils.print_extra(
                    f'Prompt: {prompt} | Model: {os.path.basename(model_path)} '
                    f'| Prediction: {prediction} | Score: {score:.4f}'
                )

                prompt_results['models'].append({
                    'model': os.path.basename(model_path),
                    'prediction': prediction,
                    'score': score
                })

            results.append(prompt_results)

        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as fp:
            json.dump(results, fp, indent=2)

        PrintUtils.print_extra(f'Inference completed. Results saved to {args.output}')

    except KeyboardInterrupt:
        PrintUtils.print_extra('Operation cancelled by user.')

    except Exception as exc:
        PrintUtils.print_error(f'Error during inference: {exc}')

    finally:
        NetworkUtils.stop_sniffing_tls(best_effort=True)
        PrintUtils.print_extra('Finished inference run.')


if __name__ == '__main__':
    main()
