#include <math.h>

#include "fourier.h"

void normalize(complex s[], int n) {
    for (int k = 0; k < n; k++) {
        s[k].a /= n;
        s[k].b /= n;
    }
}

void nft(complex s[], complex t[], int n, int sign) {
    for (int k = 0; k < n; k++) {
        t[k].a = 0;
        t[k].b = 0;

        for (int j = 0; j < n; j++) {
            double x = sign * 2 * PI * k * j / n;

            double cosx = cos(x);
            double sinx = sin(x);

            t[k].a += s[j].a * cosx - s[j].b * sinx;
            t[k].b += s[j].a * sinx + s[j].b * cosx;
        }
    }
}

void nft_forward(complex s[], complex t[], int n) {
    nft(s, t, n, -1);
}

void nft_inverse(complex t[], complex s[], int n) {
    nft(t, s, n, 1);
    normalize(s, n);
}

void fft(complex s[], complex t[], int n, int sign) {
    if (n == 1) {
        t[0].a = s[0].a;
        t[0].b = s[0].b;
        return;
    }

    int half = n / 2;
    complex sig_par[half], sig_imp[half];
    complex result_par[half], result_imp[half];

    for (int j= 0; j < half; j++) {
        sig_par[j].a = s[2 * j].a;
        sig_par[j].b = s[2 * j].b;
        sig_imp[j].a = s[2 * j + 1].a;
        sig_imp[j].b = s[2 * j + 1].b;
    }

    fft(sig_par, result_par, half, sign);
    fft(sig_imp, result_imp, half, sign);

    for (int k = 0; k < half; k++) {
        double x = (sign * 2 * PI * k) / n;
        double cosx = cos(x);
        double sinx = sin(x);

        double wa = (result_imp[k].a * cosx) - (result_imp[k].b * sinx);
        double wb = (result_imp[k].a * sinx) + (result_imp[k].b * cosx);

        t[k].a = result_par[k].a + wa;
        t[k].b = result_par[k].b + wb;
        t[k + half].a = result_par[k].a - wa;
        t[k + half].b = result_par[k].b - wb;
    }
}

void fft_forward(complex s[], complex t[], int n) {
    fft(s, t, n, -1);
}

void fft_inverse(complex t[], complex s[], int n) {
    fft(t, s, n, 1);
    normalize(s, n);
}

void fft_forward_2d(complex matrix[MAX_SIZE][MAX_SIZE], int width, int height) {
    complex col[MAX_SIZE], col_t[MAX_SIZE];
    complex linha_t[MAX_SIZE];

    for (int x= 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            col[y].a = matrix[y][x].a;
            col[y].b = matrix[y][x].b;
        }
        fft_forward(col, col_t, height);
        for (int y = 0; y < height; y++) {
            matrix[y][x].a = col_t[y].a;
            matrix[y][x].b = col_t[y].b;
        }
    }

    for (int y=0; y<height; y++) {
        fft_forward(matrix[y], linha_t, width);
        for (int x=0; x<width; x++) {
            matrix[y][x].a = linha_t[x].a;
            matrix[y][x].b = linha_t[x].b;
        }
    }
}

void fft_inverse_2d(complex matrix[MAX_SIZE][MAX_SIZE], int width, int height) {
    complex linha_t[MAX_SIZE];
    complex col[MAX_SIZE], col_t[MAX_SIZE];

    for (int y = 0; y < height; y++) {
        fft_inverse(matrix[y], linha_t, width);
        for (int x = 0; x < width; x++) {
            matrix[y][x].a = linha_t[x].a;
            matrix[y][x].b = linha_t[x].b;
        }
    }

    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            col[y].a = matrix[y][x].a;
            col[y].b = matrix[y][x].b;
        }
        fft_inverse(col, col_t, height);
        for (int y =0; y < height; y++) {
            matrix[y][x].a = col_t[y].a;
            matrix[y][x].b = col_t[y].b;
        }
    }
}

void filter(complex input[MAX_SIZE][MAX_SIZE], complex output[MAX_SIZE][MAX_SIZE], int width, int height, int flip) {
    int center_x = width / 2;
    int center_y = height / 2;

    double variance = -2 * SIGMA * SIGMA;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int diff_x = center_x - (x + center_x) % width;
            int diff_y = center_y - (y + center_y) % height;

            double dist = (diff_x * diff_x) + (diff_y * diff_y);
            double g = exp(dist / variance);

            if (flip) {
                g = 1 - g;
            }

            output[y][x].a = g * input[y][x].a;
            output[y][x].b = g * input[y][x].b;
        }
    }
}

void filter_lp(complex input[MAX_SIZE][MAX_SIZE], complex output[MAX_SIZE][MAX_SIZE], int width, int height) {
    filter(input, output, width, height, 0);
}

void filter_hp(complex input[MAX_SIZE][MAX_SIZE], complex output[MAX_SIZE][MAX_SIZE], int width, int height) {
    filter(input, output, width, height, 1);
}