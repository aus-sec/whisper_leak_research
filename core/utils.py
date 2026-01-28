import os
import colorama
import sys
import ctypes
import re
import base64
import argparse
import time
import psutil
import subprocess
import signal
import json

# Initialize colorama
colorama.init()

class ThrowingArgparse(argparse.ArgumentParser):
    """
        Custom argument parser that does not bail out or print usage on parsing errors, but throws an exception instead.
    """

    def error(self, message):
        """
            Handles errors gracefully.
        """

        # Throw
        raise Exception(message)

class OsUtils(object):
    """
        Generic OS utilities.
    """

    @staticmethod
    def del_file(path):
        """
            Tries to delete a file (best-effort).
            Indicates whether the file doesn't exist afterwards.
        """

        # Best-effort deletion
        try:
            os.unlink(path)
        except Exception:
            pass

        # Indicate result
        return not os.path.isfile(path)

    @staticmethod
    def mkdir(path):
        """
            Try to make a directory (best-effort).
            Indicates whether the directory exists afterwards.
        """

        # Best-effort creation
        try:
            os.makedirs(path, exist_ok=True)
        except Exception:
            pass

        # Indicate result
        return os.path.isdir(path)

    @staticmethod
    def is_high_privileges():
        """
            Indicates if we run in high privileges.
        """

        # Handle POSIX
        if sys.platform in ('linux', 'darwin'):
            return os.geteuid() == 0

        # Handle Windows
        if sys.platform == 'win32':
            return ctypes.windll.shell32.IsUserAnAdmin() != 0

        # Unsupported platform
        raise Exception(f'Unsupported platform "{sys.platform}"')

class PromptUtils(object):
    """
        Prompt utilities.
    """

    @staticmethod
    def read_prompts(json_path):
        """
            Reads prompts file and validate its structure.
        """

        # Read prompts
        PrintUtils.start_stage(f'Reading prompts')
        with open(json_path, 'r') as fp:
            contents = json.load(fp)

        # Validate the file structure
        assert isinstance(contents, dict), Exception('Invalid format for prompts JSON file')
        prompt_types = [ 'positive', 'negative' ]
        for prompt_type in prompt_types:
            prompts_data = contents.get(prompt_type, None)
            assert prompts_data is not None, Exception(f'Missing {prompt_type} prompts')
            assert isinstance(prompts_data, dict), Exception(f'Invalid structure for {prompt_type} prompts')
            repeats = prompts_data.get('repeat', None)
            assert repeats is not None, Exception(f'Missing key "repeat" in {prompt_type} prompts')
            assert isinstance(repeats, int) and repeats > 0, Exception(f'Invalid repeat value in {prompt_type} prompts')
            prompts = prompts_data.get('prompts', None)
            assert prompts is not None, Exception(f'Missing key "prompts" in {prompt_type} prompts')
            assert isinstance(prompts, list), Exception(f'Invalid structure for prompts in {prompts_type} prompts')
            assert len(prompts) > 0, Exception(f'The prompt list for {prompt_type} prompts is empty')
            assert len([ elem for elem in prompts if not isinstance(elem, str) ]) == 0, Exception('Invalid prompt format in {prompts_type} prompts')
            PrintUtils.print_extra(f'Loaded *{len(prompts)}* {prompt_type} prompts with repetition of *{repeats}*')

        # Return result
        PrintUtils.end_stage()
        return contents

class PrintUtils(object):
    """
        Printing utilities.
    """

    # Define colors
    WHITE = colorama.Fore.WHITE + colorama.Style.BRIGHT
    GREEN = colorama.Fore.GREEN + colorama.Style.BRIGHT
    RED = colorama.Fore.RED + colorama.Style.BRIGHT
    GREY = colorama.Fore.WHITE+ colorama.Style.NORMAL
    YELLOW = colorama.Fore.YELLOW + colorama.Style.NORMAL
    DARKGREY = colorama.Fore.LIGHTBLACK_EX + colorama.Style.BRIGHT
    RESET_COLORS = colorama.Style.RESET_ALL

    # Pretty printing
    PP_LEN = 120

    # Saves stage and extra
    _in_stage = False
    _extra = []

    @classmethod
    def print_logo(cls):
        """
            Prints the logo.
        """

        # Print the logo
        logo = base64.b64decode(b'CiAgICDilojilojilZcgICAg4paI4paI4pWX4paI4paI4pWXICDilojilojilZfilojilojilZfilojilojilojilojilojilojilojilZfilojilojilojilojilojilojilZcg4paI4paI4paI4paI4paI4paI4paI4pWX4paI4paI4paI4paI4paI4paI4pWXICAgICDilojilojilZcgICAgIOKWiOKWiOKWiOKWiOKWiOKWiOKWiOKVlyDilojilojilojilojilojilZcg4paI4paI4pWXICDilojilojilZcKICAgIOKWiOKWiOKVkSAgICDilojilojilZHilojilojilZEgIOKWiOKWiOKVkeKWiOKWiOKVkeKWiOKWiOKVlOKVkOKVkOKVkOKVkOKVneKWiOKWiOKVlOKVkOKVkOKWiOKWiOKVl+KWiOKWiOKVlOKVkOKVkOKVkOKVkOKVneKWiOKWiOKVlOKVkOKVkOKWiOKWiOKVlyAgICDilojilojilZEgICAgIOKWiOKWiOKVlOKVkOKVkOKVkOKVkOKVneKWiOKWiOKVlOKVkOKVkOKWiOKWiOKVl+KWiOKWiOKVkSDilojilojilZTilZ0KICAgIOKWiOKWiOKVkSDilojilZcg4paI4paI4pWR4paI4paI4paI4paI4paI4paI4paI4pWR4paI4paI4pWR4paI4paI4paI4paI4paI4paI4paI4pWX4paI4paI4paI4paI4paI4paI4pWU4pWd4paI4paI4paI4paI4paI4pWXICDilojilojilojilojilojilojilZTilZ0gICAg4paI4paI4pWRICAgICDilojilojilojilojilojilZcgIOKWiOKWiOKWiOKWiOKWiOKWiOKWiOKVkeKWiOKWiOKWiOKWiOKWiOKVlOKVnSAKICAgIOKWiOKWiOKVkeKWiOKWiOKWiOKVl+KWiOKWiOKVkeKWiOKWiOKVlOKVkOKVkOKWiOKWiOKVkeKWiOKWiOKVkeKVmuKVkOKVkOKVkOKVkOKWiOKWiOKVkeKWiOKWiOKVlOKVkOKVkOKVkOKVnSDilojilojilZTilZDilZDilZ0gIOKWiOKWiOKVlOKVkOKVkOKWiOKWiOKVlyAgICDilojilojilZEgICAgIOKWiOKWiOKVlOKVkOKVkOKVnSAg4paI4paI4pWU4pWQ4pWQ4paI4paI4pWR4paI4paI4pWU4pWQ4paI4paI4pWXIAogICAg4pWa4paI4paI4paI4pWU4paI4paI4paI4pWU4pWd4paI4paI4pWRICDilojilojilZHilojilojilZHilojilojilojilojilojilojilojilZHilojilojilZEgICAgIOKWiOKWiOKWiOKWiOKWiOKWiOKWiOKVl+KWiOKWiOKVkSAg4paI4paI4pWRICAgIOKWiOKWiOKWiOKWiOKWiOKWiOKWiOKVl+KWiOKWiOKWiOKWiOKWiOKWiOKWiOKVl+KWiOKWiOKVkSAg4paI4paI4pWR4paI4paI4pWRICDilojilojilZcKICAgICDilZrilZDilZDilZ3ilZrilZDilZDilZ0g4pWa4pWQ4pWdICDilZrilZDilZ3ilZrilZDilZ3ilZrilZDilZDilZDilZDilZDilZDilZ3ilZrilZDilZ0gICAgIOKVmuKVkOKVkOKVkOKVkOKVkOKVkOKVneKVmuKVkOKVnSAg4pWa4pWQ4pWdICAgIOKVmuKVkOKVkOKVkOKVkOKVkOKVkOKVneKVmuKVkOKVkOKVkOKVkOKVkOKVkOKVneKVmuKVkOKVnSAg4pWa4pWQ4pWd4pWa4pWQ4pWdICDilZrilZDilZ0KICAgICAgICAgICAgICAgIERhdGEgbGVha2FnZSBwcm9vZi1vZi1jb25jZXB0IGZvciBMYXJnZSBMYW5ndWFnZSBNb2RlbCBjaGF0Ym90cwoKICAgICAgICAgICAgICAgICAgICAgICAgICAgSm9uYXRoYW4gQmFyIE9yICgiSkJPIiksIEB5b195b195b19qYm8KICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIEdlb2ZmIE1jRG9uYWxkLCBAZ2xtY2RvbmEKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCg==').decode()
        print(f'{cls.WHITE}{logo}{cls.RESET_COLORS}')

    @classmethod
    def is_in_stage(cls):
        """
            Indicates whether we are in stage.
        """

        # Return result
        return cls._in_stage

    @classmethod
    def start_stage(cls, message, override_prev=False):
        """
            Starts a stage.
        """

        # Validate we are not in stage unless we override a previous stage possible
        #assert (not cls._in_stage) or override_prev, Exception('Entering a stage without finishing the previous one')
        if cls._in_stage and not override_prev:
            # End the stage
            cls.end_stage()

        # Potentially override
        if override_prev:
            print('\r' + (' ' * cls.PP_LEN) + '\r', end='')

        # Print title
        title = message[:cls.PP_LEN - 5]
        title += ' ...'
        title += (cls.PP_LEN - 1 - len(title)) * '.'
        title += ' '
        print(f'{cls.GREY}{title}{cls.RESET_COLORS}', end='')

        # Flush
        sys.stdout.flush()

        # Indicate we are in stage
        cls._in_stage = True

    @classmethod
    def end_stage(cls, fail_message=None, throw_on_fail=True):
        """
            Ends a stage.
        """

        # Print status
        status = f'{cls.GREY}[  {cls.GREEN}OK{cls.GREY}  ]{cls.RESET_COLORS}' if fail_message is None else f'{cls.GREY}[ {cls.RED}FAIL{cls.GREY} ]{cls.RESET_COLORS}'
        print(status)

        # Indicate we are not in a stage
        cls._in_stage = False

        # Prints extra contents
        cls._dump_extra()

        # Optionally throw
        if throw_on_fail and (fail_message is not None):
            raise Exception(fail_message)

    @classmethod
    def print_error(cls, message):
        """
            Prints a raw error message.
        """

        # Prints the error message
        print(f'{cls.RED}ERROR{cls.GREY}: {message}{cls.RESET_COLORS}')

    @classmethod
    def print_extra(cls, message):
        """
            Prints extra contents. Will be pending if stage is not complete.
        """

        # Adds message as an extra
        cls._extra.append(message)
        if not cls._in_stage:
            cls._dump_extra()

    @classmethod
    def _dump_extra(cls):
        """
            Dumps extra contents.
        """

        # Validate we are not in a stage
        assert not cls._in_stage, Exception('Cannot dump extra while in a stage')
        
        # Print messages and replace special strings with pretty colors
        for message in cls._extra:
            msg = re.sub(r'\*.+?\*', lambda m:f'{cls.YELLOW}{m.group(0)[1:-1]}{cls.DARKGREY}', message)
            print(f'{cls.DARKGREY}{msg}{cls.RESET_COLORS}')
        
        # Reset extra
        cls._extra = []

class NetworkUtils:
    """
        Network utilities with cross-platform support for packet capture.
    """

    # Sniffing handle
    _sniffer = None
    _capture_file = None

    @staticmethod
    def get_self_local_ports(remote_port):
        """
            Gets local TCP ports by self process connected to the given remote port.
            Return the set of local port numbers with established connections
        """

        # Returns all local ports
        return set([ conn.laddr.port for conn in psutil.net_connections(kind='inet') if (getattr(conn.raddr, 'port', None) == remote_port and conn.status == 'ESTABLISHED' and conn.pid == os.getpid()) ])

    @classmethod
    def start_sniffing_tls(cls, pcap_file_path, remote_port=443):
        """
            Starts sniffing TLS traffic.
        """

        # Validate sniffing is not being done
        if cls._sniffer is not None:
            raise Exception('Active sniffing already in progress')

        # Save the capture file
        cls._capture_file = pcap_file_path
        
        # Handle POSIX
        if sys.platform in ('linux', 'darwin'):
        
            # Unix-based systems use tcpdump
            cls._sniffer = subprocess.Popen([ 'tcpdump', '-w', pcap_file_path, f'tcp port {remote_port}' ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif sys.platform == 'win32':

            # For Windows, use pktmon (we first reset any existing filters)
            subprocess.run([ 'pktmon', 'filter', 'remove' ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Add TCP filter for the specified port
            subprocess.run([ 'pktmon', 'filter', 'add', '-t', 'tcp', '-p', str(remote_port) ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Start capture with the filtered port
            cls._sniffer = subprocess.Popen([ 'pktmon', 'start', '-c', '-f', pcap_file_path + '.etl' ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            raise Exception(f'Unsupported platform: "{sys.platform}"')
            
        # Wait for capture to initialize
        time.sleep(2)

    @classmethod
    def stop_sniffing_tls(cls, best_effort=False):
        """
            Stops sniffing TLS traffic.
        """

        # Validate sniffing is being done
        if cls._sniffer is None:
            if best_effort:
                return
            raise Exception('No active sniffing has been started')

        # Allow packets to finish processing
        time.sleep(2)
       
        # Handle POSIX
        if sys.platform in ('linux', 'darwin'):

            # Send a SIGINT and wait
            cls._sniffer.send_signal(signal.SIGINT)
            cls._sniffer.wait(timeout=5)

        # Handle Windows
        elif sys.platform == 'win32':

            # Windows: stop pktmon and convert ETL to PCAP
            subprocess.run([ 'pktmon', 'stop' ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Convert the ETL file to PCAP format
            etl_file = cls._capture_file + '.etl'
            subprocess.run([ 'pktmon', 'etl2pcap', etl_file, '--out', cls._capture_file ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # Validate PCAP exists
            assert os.path.isfile(cls._capture_file), Exception('Conversion from ETL to PCAP failed')
            
            # Clean up the ETL file
            OsUtils.del_file(etl_file)
            
            # Remove the filters
            subprocess.run([ 'pktmon', 'filter', 'remove' ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Nullify members
        cls._sniffer = None
        cls._capture_file = None

