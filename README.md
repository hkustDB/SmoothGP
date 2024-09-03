# Smooth Sensitivity for Geo-Privacy
Yuting Liang and Ke Yi. 2024. Smooth Sensitivity for Geo-Privacy. In Proceedings of the 2024 ACM SIGSAC Conference on Computer and Communications Security (CCS ’24)

The experiments were implemented in Python v3.10.
### Dependencies
numpy v1.23.5 <br/>
scipy v1.9.3 <br/>

### Data
The CENSUS dataset in the ./data folder has been compressed. Please de-compress prior to running the experiments.<br/>
The New York motor vehicle collision dataset can be downloaded from the official website provided in the paper.

### Secure random number generation
Implementations for the mechanisms which use a cryptographically secure random number generator have been added (functions suffixed with "_secure"). Note that for testing purposes, the regular implementations which use numpy for random number generation should be used, since they are more efficient.
