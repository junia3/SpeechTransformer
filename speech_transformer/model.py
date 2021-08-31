import torch.nn as nn

from speech_transformer.tfmodules import SpeechTransformerDecoder, SpeechTransformerEncoder

class  SpeechTransformer(nn.Module):
    def __init__(
            self,
            num_classes = 45,
            d_model = 256,
            input_dim = 94,
            pad_id = 40,
            sos_id = 43,
            eos_id = 44,
            d_ff = 1024,
            num_heads = 4,
            num_encoder_layers = 6,
            num_decoder_layers = 6,
            dropout_p = 0.3,
            ffnet_style = 'ff',
            max_length= 128,
    ):
        super(SpeechTransformer, self).__init__()
        assert d_model % num_heads == 0, "d_model % num_heads should be zero."

        self.num_classes = num_classes
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.max_length = max_length
        
        self.encoder = SpeechTransformerEncoder(
            d_model=d_model,
            input_dim=input_dim,
            d_ff=d_ff,
            num_layers=num_encoder_layers,
            num_heads=num_heads,
            ffnet_style=ffnet_style,
            dropout_p=dropout_p,
            pad_id=pad_id,
        )

        self.decoder = SpeechTransformerDecoder(
            num_classes=num_classes,
            d_model=d_model,
            d_ff=d_ff,
            num_layers=num_decoder_layers,
            num_heads=num_heads,
            ffnet_style=ffnet_style,
            dropout_p=dropout_p,
            pad_id=pad_id,
            sos_id=sos_id,
            eos_id=eos_id,
        )

    def forward(
            self,
            inputs,
            input_lengths,
            targets = None,
            target_lengths = None,
    ):
        logits = None
        encoder_outputs, encoder_output_lengths = self.encoder(inputs, input_lengths)
        logits = self.decoder(
            encoder_outputs=encoder_outputs,
            encoder_output_lengths=encoder_output_lengths,
            targets=targets,
            target_lengths=target_lengths,
        ) ## log softmax score
        predictions = logits.max(-1)[1]  ## prediction for label

        return predictions, logits
