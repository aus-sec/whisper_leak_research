#!/usr/bin/env python3
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

from core.utils import PrintUtils
from core.utils import OsUtils
from core.utils import ThrowingArgparse
from core.utils import NetworkUtils
from core.chatbot_utils import ChatbotUtils
from core.model import Sequence
from core.classifiers.base_classifier import BaseClassifier

import pyshark
import os
import json
import torch
import sys
import sys

# Graph
GRAPH_BLOCKS = '▁▂▃▄▅▆▇█'

# Global - the chatbot object
g_chatbot_object = None

# Global - stream sequences
g_stream_sequences = {}

# Global - the model object and device
g_model_object = None
g_device = None

def get_self_dir():
    """
        Get the self directory.
    """

    # Return the self directory
    return os.path.dirname(os.path.abspath(__file__))

def parse_arguments():
    """
        Parse arguments.
    """

    # Parsing arguments
    PrintUtils.start_stage('Parsing command-line arguments')
    parser = ThrowingArgparse()
    parser.add_argument('-i', '--interface', help='The network interface', required=True)
    parser.add_argument('-c', '--chatbot', help='The chatbot', required=True)
    parser.add_argument('-m', '--model', help='The model path', required=True)

    args = parser.parse_args()
    PrintUtils.end_stage()

    # Return the parsed arguments
    return args

def get_chatbot_object(chatbot_name):
    """
        Get the chatbot object from the given chatbot name.
    """

    # Load all chatbots
    PrintUtils.start_stage('Loading chatbots')
    chatbots = ChatbotUtils.load_chatbots(os.path.join(get_self_dir(), 'chatbots'))
    assert len(chatbots) > 0, Exception('Could not load any chatbots')
    chatbot_names = '\n'.join([ f'\t*{chatbot.__name__}*' for chatbot in chatbots.values() ])
    PrintUtils.print_extra(f'Loaded chatbots:\n{chatbot_names}')
    PrintUtils.end_stage()

    # Validating chatbot class exists
    PrintUtils.start_stage('Initializing chatbot class')
    chatbot_class = chatbots.get(chatbot_name.lower(), None)
    assert chatbot_class is not None, Exception(f'Chatbot "{chatbot_name}" does not exist')
    PrintUtils.print_extra(f'Using chatbot *{chatbot_class.__name__}*')
    PrintUtils.end_stage()

    # Return the object
    return chatbot_class()

def packet_callback(packet):
    """
        Callback for all packets.
    """

    # Using the stream sequences
    global g_stream_sequences

    # Handle gracefully
    try:

        # Only handle TLS packets
        if not hasattr(packet, 'tls'):
            return

        # Support both IPv4 and IPv6
        addr = packet.ip if hasattr(packet, 'ip') else packet.ipv6

        # Handle handleshakes to identify streams to follow
        if getattr(packet.tls, 'handshake_type', None) == '1':
            server_name = getattr(packet.tls, 'handshake_extensions_server_name', None)
            if server_name is not None and g_chatbot_object.match_tls_server_name(server_name):
                g_stream_sequences[(addr.dst, int(packet.tcp.dstport), addr.src, int(packet.tcp.srcport))] = Sequence(float(packet.sniff_time.timestamp()))   # Note we saved the reverse 4-tuple due to interest in incoming data
                PrintUtils.print_extra(f'\n{packet.sniff_time}: New stream: *{addr.src}:{packet.tcp.srcport}* --> *{addr.dst}:{packet.tcp.dstport}*')

        # Handle Application Data
        if hasattr(packet.tls, 'app_data'):

            # Only handle streams to follow
            key = (addr.src, int(packet.tcp.srcport), addr.dst, int(packet.tcp.dstport))
            sequence = g_stream_sequences.get(key, None)
            if sequence is None:
                return

            # Follow sequence
            timestamp = float(packet.sniff_time.timestamp())
            data_length = int(packet.length)
            sequence.add_pair(timestamp, data_length)

            # Infer from model
            prob, pred = g_model_object.inference((sequence.time_seq, sequence.size_seq), g_device)
            msg = f'{PrintUtils.RED}!' if pred > 0.9 else f'{PrintUtils.GREEN}V'
            print(f'\b\b{GRAPH_BLOCKS[int(prob * (len(GRAPH_BLOCKS) - 1))]} {msg}{PrintUtils.RESET_COLORS}', end='')
            sys.stdout.flush()

    # Log exceptions
    except Exception as ex:
        PrintUtils.print_extra(f'*WARNING* - error processing packet: {ex}')

def main():
    """
        Main routine.
    """

    # Using globals
    global g_chatbot_object
    global g_model_object
    global g_device

    # Catch-all
    is_user_cancelled = False
    last_error = None
    capture = None
    try:

        # Print logo
        PrintUtils.print_logo()

        # Validate high privileges
        PrintUtils.start_stage('Validating high privileges')
        assert OsUtils.is_high_privileges(), Exception('User does not run in high privileges')
        PrintUtils.end_stage()

        # Parsing arguments
        args = parse_arguments()

        # Setting device
        PrintUtils.start_stage('Setting device')
        g_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        PrintUtils.end_stage()
        PrintUtils.print_extra(f'Using device: *{g_device}*')

        # Get the chatbot object
        g_chatbot_object = get_chatbot_object(args.chatbot)

        # Get the model object
        PrintUtils.start_stage('Loading model')
        g_model_object = BaseClassifier.load(args.model, g_device)
        PrintUtils.end_stage()

        # Start the capture
        PrintUtils.start_stage('Initializing sniffing')
        capture = pyshark.LiveCapture(interface=args.interface, display_filter='tcp')
        PrintUtils.end_stage()
        capture.apply_on_packets(packet_callback)

    # Handle exceptions
    except Exception as ex:

        # Optionally fail stage
        if PrintUtils.is_in_stage():
            PrintUtils.end_stage(fail_message=ex, throw_on_fail=False)

        # Save error and print it as an extra
        PrintUtils.print_extra(f'Error: {ex}')
        last_error = ex

    # Handle cancel operations
    except KeyboardInterrupt:
        if PrintUtils.is_in_stage():
            PrintUtils.end_stage(fail_message='', throw_on_fail=False)
        PrintUtils.print_extra(f'Operation *cancelled* by user - please wait for cleanup code to complete')
        is_user_cancelled = True

    # Cleanups
    finally:

        # Cleanup any sniffing that may still be happening
        PrintUtils.start_stage('Running cleanup code')
        if capture is not None:
            capture.close()
        PrintUtils.end_stage()

        # Print final status
        if last_error is not None:
            PrintUtils.print_error(f'{last_error}\n')
        elif is_user_cancelled:
            PrintUtils.print_extra(f'Operation *cancelled* by user\n')
        else:
            PrintUtils.print_extra(f'Finished successfully\n')

if __name__ == '__main__':
    main()
