## 28.03.2023
- zapisać pliki do h5py
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

## 12.03.2023
- zrobić wykres spektogram(różnica względem poprzedniej próbki)
- spektogram prędkości obieku
- spróbować rozpoznawać obiekt żywny - nie
- maksymalizacja energii do wykrywania obiektów

pytania:
czy fft odległości powinno być liczbą zespoloną czy normalną

## 18.04.2023

1 fft po wymiarze chirp
2 fft po wymiarze bin
3 fft w stosunkowo malym oknie, okno gdzie mozna zalozyc ze predkosc jest stala

sygnal roznicowy liczyc w czasie, nie w czestotliwosci:

diff_resample = signal_resample[1:] - signal_resample[:-1]

sprawdzic rozne spektra roznicowe: co 1/10/100/1000 ms, sprawdzic co widac, jaka roznica czasu lepsza?
poprawne plotowanie w skali logarytmicznej
algorytm wykrywania -> sledzenia

wykres odleglosci od czasu, na nim trajektoria

wtorek 16:15

## 25.04.2023

1. w otoczeniu punktu nadal jest wysoka wartosc
2. programowanie dynamiczne, maksymalizacja energii
3. algorytm progowania OTSU
4. okna do svm mogą się nachodzić w 50%













