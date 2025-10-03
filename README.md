# Genre-and-Mood-Conditioned-Music-Generator
This notebook's generative AI system produces genre and mood-conditioned REMI tokenised sequences. These can be detokenized into MIDI files and then synthesised into WAV files using a SoundFont (Fluidr3_gm.sf2). 

### Dataset:
The MIDI files are obtained from Lakh MIDI.
The genres and moods are obtained from MidiCaps, a descriptive dataset on 168,385 of these MIDIs.

### Data Preparation:
The number of tracks and tokens used is chosen.
Then, these files are tokenised using the REMI tokeniser from MidiTok and prepended by special tokens that indicate the genres and moods of that track. They are then saved into
a .npy file.

### Training:
These tokens are passed to an autoregressive transformer built in PyTorch. The default model has been trained on the first 1,000 tokens of 25,000 tracks from the above dataset and is constructed to predict the next token in the sequence based on all previous tokens.

### Generation:
The model is seeded with the user-defined special mood and genre tokens. Based on the
previous tokens, it generates one token at a time until a complete musical sequence is
generated.

### Motivation:
This project is motivated by a desire to create a music generation system capable of generating custom music for those who need it, like indie game developers or small content creators. The flexible conditioning system is designed to make music generation controllable for their specific needs. In the future, I would like to refine the project to create music that meets professional production standards, as it is not yet at that level.

### Details:
The primary generative system in this project is an autoregressive transformer, which is implemented in PyTorch. Although the model is built with a Transformer Encoder layer, it is designed to behave like a decoder through causal attention masking. This limits the model to only access the current and previous tokens during training and generation, forcing it to act like an autoregressive decoder, generating sequences one token at a time.
Before this, embedding layers create token and positional embeddings, enabling the transformer to understand the tokens and their order. Following the transformer layer, normalisation is carried out, followed by the token prediction.
During training, the model is given every token before position x as the data, and the token at position x as the target, training the model to generate what it thinks should be
the following token.
The model's training input is a sequence of tokens of a specified length. By default, the sequence length is 1024 – 12 genre and 12 mood tokens, followed by 1000 REMI tokens.
The output is the next token, based on the current tokens. The provided model's training data consisted of the first 1000 tokens from 25000 songs in the Lakh MIDI dataset, although the user can also choose these if they decide to train the model.
