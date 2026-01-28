import boto3
import json
import os
from dotenv import load_dotenv
from core.chatbot_utils import ChatbotBase, LocalPortSaverTransport

class AmazonNovaLiteV1(ChatbotBase):
    """
        Amazon Nova chatbot utilizing streaming responses.
    """

    def __init__(self, remote_tls_port=443):
        """
            Initializes the chatbot instance.
        """
        
        # Initialize
        super().__init__(remote_tls_port)

        # Load environment variables from .env file
        load_dotenv()

        # Retrieve AWS credentials from environment variables
        aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        aws_session_token = os.getenv('AWS_SESSION_TOKEN')  # Optional

        # Check if any required credentials are missing
        if not aws_access_key_id or not aws_secret_access_key:
            raise ValueError('AWS credentials are not set in the environment variables. Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY.')

        # Initialize the Boto3 client
        self._transport = LocalPortSaverTransport()
        self._client = boto3.client(
            'bedrock-runtime',
            region_name='us-east-1',
            aws_access_key_id=str(aws_access_key_id),
            aws_secret_access_key=str(aws_secret_access_key)
        )
        self._model_id = 'us.amazon.nova-lite-v1:0'

    def send_prompt(self, prompt, temperature):
        """
            Sends a prompt to the model and returns the response along with the local port.
        """

        message_list = [{ 'role': 'user', 'content': [{ 'text': prompt }]}]
        inf_params = { 'maxTokens': 3000, 'temperature': temperature}
        request_body = {
            'schemaVersion': 'messages-v1',
            'messages': message_list,
            'system': [{'text': ''}],
            'inferenceConfig': inf_params,
        }

        result = self._client.invoke_model_with_response_stream(
            modelId=self._model_id, body=json.dumps(request_body)
        )

        response = []

        stream = result.get('body')
        if stream:
            for event in stream:
                chunk = event.get('chunk')
                if chunk:
                    chunk_json = json.loads(chunk.get('bytes').decode())
                    content_block_delta = chunk_json.get('contentBlockDelta')
                    if content_block_delta:
                        text_chunk = content_block_delta.get('delta', {}).get('text', '')
                        response.append(text_chunk)

        return response, self._transport.get_local_port()

    def get_temperature(self):
        """
            Returns the default temperature setting for the model.
        """

        # Return 1.0 by default
        return 1.0

    def get_common_name(self):
        """
            Gets the common name of the model.
        """

        # Return common name
        return 'nova-light-v1'

    def match_tls_server_name(self, server_name):
        """
            Matches the TLS server name.
        """

        # Match
        return 'amazonaws.com' in server_name and 'bedrock-runtime' in server_name
