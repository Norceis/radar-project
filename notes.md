## 28.03.2023
- zapisać pliki do hdfs? (h5py)
    - można je wczytywać po kawałku
    - albo co x ramek
    - albo od jakiegoś punktu
    - powinien ważyć tyle co bin, zoabczymy 

- w funkcji do pobierania danych, 2 argument oznacza ile anten było wykorzystywana 
    - pilnować czy ilość klatek zgadza się z czasem nagrania

- kąt wynika z triangulacji, czegoś, pewnie sygnałów z dwóch anten

- oś poziomą przeliczyć na metry, jest ona podana w binach(512). scipy.fft -> fftfreqs, ilość binów i częstotliwość próbkowania(w configu, 2924kHz). Uzyskane częstotliwości przeliczyć na metry. Potrzebny do tego frequency slope

- wykres fft dla kilku czerpów po kolei, najlepiej jak ktoś chodzi

- następne spotkanie 12.04, między 15-16, stacjonarnie



https://github.com/0xastro/fmcw-RADAR

http://radar.alizadeh.ca/introduction.html#range-detection

https://github.com/PreSenseRadar/OpenRadar/blob/master/Presense%20Applied%20Radar/basics/Range%20-%20COMPLETED.ipynb

## 12.03.2023
- zrobić wykres spektogram(różnica względem poprzedniej próbki)
- spektogram prędkości obieku
- spróbować rozpoznawać obiekt żywny - nie
- maksymalizacja energii do wykrywania obiektów