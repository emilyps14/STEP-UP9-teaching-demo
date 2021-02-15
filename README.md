# STEP-UP9-teaching-demo

Teaching demo on Multitaper Spectral Estimation

Slides: 
- https://docs.google.com/presentation/d/16NmZTfpbiZLKb73zx5UUpZK16JEaflHTMcFAYx94e98/edit?usp=sharing
- https://docs.google.com/presentation/d/1D-z75VaJqD0aF-3twjMQlKl9jOwB4MaVq00fWQwjYN0/edit?usp=sharing

## Activity Breakdown
Prereqs: Computing Spectrograms using windowed Fourier Transform. Time/frequency tradeoff and the effects of different windowing functions (filters in the time and frequency domain). Main lobe vs side lobe. 

2 mins: Lecture. Slepian Tapers as orthogonal windowing functions, minimizing side lobes at the expense of main lobe width. Multitaper spectrograms: time/frequency/statistical power tradeoff in parameters T (time resolution), W (half-bandwidth), K (# tapers).

2 mins (Formative Assessment): Quiz: Matching. Match the TWK parameters with their time domain and frequency domain slepian tapers. (1 minute to think, 1 minute to tell them the answer) As a class: which set will give you the best time resolution? frequency resolution? statistical power?

1 min: Lecture: Show what delta function and sine wave look like in spectrograms using each of the three sets of parameters

10 mins (Explore/Explain, Formative Assessment): Given a timeseries, draw out what you think the spectrogram will look like for each of the three sets of parameters
Examples of: delta function, wavelet, sine wave, pulse of white noise, spectral leakage

## HOCS Learning Goal/Outcome
Application: Choose parameters to use in Multitaper analysis in different scientific contexts

## Diagnostic/Formative Assessment
The drawing activity tests the full chain of their understanding of spectral analysis, the meaning of time and frequency resolution, and the reasons why we would want multiple tapers. I'm expecting students to get the basic drawings mostly correct, but to struggle with where to look for spectral leakage and how the spectrogram would change with fewer tapers.
