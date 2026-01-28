#!/usr/bin/env python3
from core.utils import PrintUtils
from core.utils import OsUtils
from core.utils import ThrowingArgparse
from core.chatbot_utils import ChatbotUtils

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

    # Parsing arguments
    PrintUtils.start_stage('Parsing command-line arguments')
    parser = ThrowingArgparse()
    parser.add_argument('-c', '--chatbot', help='The chatbot', required=True)
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

def main():
    """
        Main routine.
    """

    # Catch-all
    last_error = None
    capture = None
    try:

        # Print logo
        PrintUtils.print_logo()

        # Parsing arguments
        args = parse_arguments()

        # Get the chatbot object
        chatbot = get_chatbot_object(args.chatbot)

        # Get prompts forever
        PrintUtils.print_extra('Press *CTRL+C* to finish')
        while True:
            prompt = input('Prompt? ').strip()
            if len(prompt) == 0:
                continue
            response = ''.join(chatbot.send_prompt(prompt, chatbot.get_temperature())[0])
            PrintUtils.print_extra(f'*Response*: {response}')

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

    # Cleanups
    finally:

        # Print final status
        if last_error is not None:
            PrintUtils.print_error(f'{last_error}\n')
        else:
            PrintUtils.print_extra(f'Finished successfully\n')

if __name__ == '__main__':
    main()
