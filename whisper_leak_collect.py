#!/usr/bin/env python3
import sys
from core.utils import PrintUtils
from core.utils import OsUtils
from core.utils import ThrowingArgparse
from core.utils import NetworkUtils
from core.utils import PromptUtils
from core.chatbot_utils import ChatbotUtils
from core.model import TrainingSetCollector

import os



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

    PrintUtils.start_stage('Parsing command-line arguments')
    parser = ThrowingArgparse()
    parser.add_argument('-c', '--chatbot', help='The chatbot to collect from.', default="AzureGPT41")
    parser.add_argument('-p', '--prompts', help='The prompts JSON file path.', default="./prompts/standard/prompts.json")
    parser.add_argument('-t', '--tlsport', type=int, help='The remote TLS port to sniff.', default=443)
    parser.add_argument('-o', '--output', type=str, help='The output folder for collected data.', default="data/main")
    parser.add_argument('-T', '--temperature', type=float, help='Override temperature value to use for the chatbot.')

    args = parser.parse_args()

    if not args.chatbot:
        parser.error("--chatbot is required for collection")
    if not args.prompts:
        parser.error("--prompts is required for collection")
    assert 0 < args.tlsport <= 0xFFFF, Exception(f'Invalid remote TLS port: {args.tlsport}')

    PrintUtils.end_stage()
    return args

def get_chatbot_class(chatbot_name):
    """
        Get the chatbot class from the given chatbot name.
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

    # Return the class
    return chatbot_class

def main():
    """
        Main routine.
    """
    is_user_cancelled = False
    last_error = None
    aggregated_entries = None
    collector = None
    chatbot_class = None

    try:
        PrintUtils.print_logo()

        args = parse_arguments()

        PrintUtils.start_stage('Validating high privileges')
        assert OsUtils.is_high_privileges(), Exception('User does not run in high privileges')
        PrintUtils.end_stage()

        PrintUtils.print_extra("Starting data collection task...")
        chatbot_class = get_chatbot_class(args.chatbot)

        prompts = PromptUtils.read_prompts(args.prompts)

        training_set_path = os.path.join(get_self_dir(), args.output)
        collector = TrainingSetCollector(
            prompts['positive']['prompts'],
            prompts['positive']['repeat'],
            prompts['negative']['prompts'],
            prompts['negative']['repeat'],
            training_set_path,
            args.tlsport
        )
        aggregated_entries = collector.get_training_set(chatbot_class, args.temperature)

        dataset_path = collector.get_dataset_path(chatbot_class.__name__, args.temperature)
        dataset_rel = os.path.relpath(dataset_path, os.getcwd())
        PrintUtils.print_extra(
            f'Aggregated dataset saved to *{dataset_rel}* with *{len(aggregated_entries)}* entries.'
        )
        PrintUtils.print_extra("Data collection task finished.")

    except KeyboardInterrupt:
        if PrintUtils.is_in_stage():
            PrintUtils.end_stage(fail_message='', throw_on_fail=False)
        PrintUtils.print_extra('Operation *cancelled* by user - please wait for cleanup code to complete')
        is_user_cancelled = True
    except Exception as ex:
        if PrintUtils.is_in_stage():
            PrintUtils.end_stage(fail_message=ex, throw_on_fail=False)
        PrintUtils.print_extra(f'Error: {ex}')
        last_error = ex
    finally:
        PrintUtils.start_stage('Running cleanup code')
        NetworkUtils.stop_sniffing_tls(best_effort=True)
        PrintUtils.end_stage()

        if last_error is not None:
            PrintUtils.print_error(f'{last_error}\n')
            sys.exit(1)
        elif is_user_cancelled:
            PrintUtils.print_extra('Operation *cancelled* by user\n')
            sys.exit(1)
        else:
            if aggregated_entries is not None:
                PrintUtils.print_extra(f'Total aggregated entries: *{len(aggregated_entries)}*')
            PrintUtils.print_extra('Collection finished successfully\n')
            sys.exit(0)

if __name__ == '__main__':
    main()
