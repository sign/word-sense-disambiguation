# Word Sense Disambiguation (WSD)

A Python implementation of word sense disambiguation using ModernBERT and spaCy for natural language processing.
This tool identifies the correct meaning of ambiguous words in context by leveraging dictionary definitions and
transformer-based language models.

This is designed for synthetic data generation, and can run at about 1 sentence per second.
If this needs to be done in production, a word-vector search approach would be much more efficient.

![Example of word sense disambiguation](assets/wsd-example.png)

## Possible Improvements

- [ ] For every definition, also include hypernym and other forms for the same synset
- [ ] Automatically search for a better prompt over a smaller benchmark dataset
- [ ] Use the benchmark dataset as a training dataset, and fine tune the model (with shuffled definitions).
- [ ] Also search for noun phrases like "bus driver" or "bass player" and disambiguate the whole phrase

## Example

For the sentence:
> The bass player adjusted the bass on his amplifier while fishing for bass.

For each content word, we generate a prompt, such as:

```txt
Instruction: 
Disambiguate the meaning of the highlighted word based on its usage in the sentence. 
Choose the most appropriate sense from the list.

Sentence: The *bass* player adjusted the bass on his amplifier while fishing for bass .

Question:
Which sense best matches the meaning of the bolded word as used in this sentence?

Choices:
- A: the lowest part of the musical range
- B: the lowest part in polyphonic music
- C: an adult male singer with the lowest voice
- D: the lean flesh of a saltwater fish of the family Serranidae
- E: any of various North American freshwater fish with lean flesh (especially of the genus Micropterus)
- F: the lowest adult male singing voice
- G: the member with the lowest range of a family of musical instruments
- H: nontechnical name for any of numerous edible marine and freshwater spiny-finned fishes

Answer: [unused0] [MASK]
```

The results show reasonable word sense disambiguation:

- "bass" (position 1): "the lowest part of the musical range" (22% confidence) ✓ Correct - musical context
- "player": "someone who plays a musical instrument (as a profession)" (81% confidence) ✓ Correct - high confidence
- "adjusted": "alter or regulate so as to achieve accuracy or conform to a standard" (48% confidence) ✓ Correct -
  moderate confidence
- "bass" (position 5): "the lowest part of the musical range" (44% confidence) ✓ Correct - audio equipment context
- "amplifier": "electronic equipment that increases strength of signals passing through it" (70% confidence) ✓ Correct -
  strong confidence
- "fishing": "catch or try to catch fish or shellfish" (87% confidence) ✓ Correct - very high confidence
- "bass" (position 12): "the lowest part of the musical range" (30% confidence) ✗ Wrong - should be fish in fishing
  context

## Benchmark

Benchmarking helps us understand whether a new model, prompt, or any change improves or
hurts the performance of the method.

In `benchmark.py`, we collect all non-trivial examples from the English WordNet -
cases where a word form has multiple possible meanings and appears in the example text.
We automatically mark target words with asterisks (*word*) in the example sentences.

For each example, we perform WSD given the marked sentence, the lemma, and the part-of-speech tag,
then compare the predicted synset ID against the ground truth.

| Device         | Model      | Time     | Accuracy |
|----------------|------------|----------|----------|
| Macbook Pro M4 | ModernBERT | 00:36:16 | 54.0%    |

50% accuracy might seem bad. However, remember we only run it on non-trivial cases, and we expect 
the language distribution to be uniform. In a real test (like the above `bass` sentence, it performs a lot better).

## More Examples

#### the big brown fox jumps over the lazy dog

- "big": "above average in size or number or quantity or magnitude or extent" (46% confidence) ✓ Correct - size descriptor
- "brown": "(of skin) deeply suntanned" (55% confidence) ✗ Wrong - should be color, not skin tone
- "fox": "alert carnivorous mammal with pointed muzzle and ears and a bushy tail; most are predators that do not hunt in packs" (84% confidence) ✓ Correct - high confidence animal identification
- "jumps": "move or jump suddenly, as if in surprise or alarm" (26% confidence) ✗ Wrong - should be physical leaping action, not startled movement
- "lazy": "disinclined to work or exertion" (98% confidence) ✓ Correct - very high confidence
- "dog": "a member of the genus Canis (probably descended from the common wolf) that has been domesticated by man since prehistoric times; occurs in many breeds" (69% confidence) ✓ Correct - strong confidence animal identification

#### Nagish captions your calls and empowers you to communicate using text or voice. It's fast, private, and accurate.

- "captions": "translation of foreign dialogue of a movie or TV program; usually displayed at the bottom of the screen" (61% confidence) ✓ Correct - subtitle/text display context
- "calls": "none of the above" (18% confidence) ✗ Wrong - should be phone calls/communication, low confidence suggests difficulty with proper nouns in context
- "empowers": "give or delegate power or authority to" (83% confidence) ✓ Correct - high confidence enablement meaning
- "communicate": "be in verbal contact; interchange information or ideas" (50% confidence) ✓ Correct - moderate confidence information exchange
- "using": "put into service; make work or employ for a particular purpose or for its inherent or natural purpose" (43% confidence) ✓ Correct - moderate confidence utilization meaning
- "text": "the words of something written" (50% confidence) ✓ Correct - moderate confidence written communication
- "voice": "the sound made by the vibration of vocal folds modified by the resonance of the vocal tract" (34% confidence) ✓ Correct - low confidence but correct vocal sound meaning
- "fast": "(used of timepieces) indicating a time ahead of or later than the correct time" (59% confidence) ✗ Wrong - should be speed/quick, not time accuracy
- "private": "confined to particular persons or groups or providing privacy" (41% confidence) ✓ Correct - moderate confidence privacy meaning
- "accurate": "conforming exactly or almost exactly to fact or to a standard or performing with total accuracy" (61% confidence) ✓ Correct - good confidence precision meaning
