/* Narrow ctypes bridge for MIRACL Core SHA-256 and AES-GCM. */
#include <string.h>
#include "core.h"

#ifdef _WIN32
#define MIRACL_EXPORT __declspec(dllexport)
#else
#define MIRACL_EXPORT
#endif

/* aes.c references this through its CBC helpers; keeping it here avoids
 * linking the otherwise-unneeded MIRACL random API. */
void OCT_clear(octet *octets) {
    if (octets == NULL) return;
    if (octets->val != NULL && octets->len > 0) {
        memset(octets->val, 0, (size_t)octets->len);
    }
    octets->len = 0;
}

MIRACL_EXPORT void proxyfl_miracl_sha256(
    const unsigned char *data, int data_len, unsigned char output[32]) {
    hash256 hash;
    int index;
    HASH256_init(&hash);
    for (index = 0; index < data_len; index++) HASH256_process(&hash, data[index]);
    HASH256_hash(&hash, (char *)output);
}

static int valid_gcm_input(int key_len, int iv_len, int aad_len, int payload_len) {
    return (key_len == 16 || key_len == 24 || key_len == 32)
        && iv_len > 0 && aad_len >= 0 && payload_len >= 0;
}

static int constant_time_equal(const unsigned char *left, const unsigned char *right) {
    unsigned char difference = 0;
    int index;
    for (index = 0; index < 16; index++) difference |= left[index] ^ right[index];
    return difference == 0;
}

MIRACL_EXPORT int proxyfl_miracl_gcm_encrypt(
    const unsigned char *key, int key_len, const unsigned char *iv, int iv_len,
    const unsigned char *aad, int aad_len, const unsigned char *plaintext, int plaintext_len,
    unsigned char *ciphertext, unsigned char tag[16]) {
    octet K = {key_len, key_len, (char *)key};
    octet IV = {iv_len, iv_len, (char *)iv};
    octet H = {aad_len, aad_len, (char *)aad};
    octet P = {plaintext_len, plaintext_len, (char *)plaintext};
    octet C = {0, plaintext_len, (char *)ciphertext};
    octet T = {0, 16, (char *)tag};
    if (!valid_gcm_input(key_len, iv_len, aad_len, plaintext_len)) return 0;
    AES_GCM_ENCRYPT(&K, &IV, &H, &P, &C, &T);
    return C.len == plaintext_len && T.len == 16;
}

MIRACL_EXPORT int proxyfl_miracl_gcm_decrypt(
    const unsigned char *key, int key_len, const unsigned char *iv, int iv_len,
    const unsigned char *aad, int aad_len, const unsigned char *ciphertext, int ciphertext_len,
    const unsigned char expected_tag[16], unsigned char *plaintext) {
    unsigned char calculated_tag[16];
    octet K = {key_len, key_len, (char *)key};
    octet IV = {iv_len, iv_len, (char *)iv};
    octet H = {aad_len, aad_len, (char *)aad};
    octet C = {ciphertext_len, ciphertext_len, (char *)ciphertext};
    octet P = {0, ciphertext_len, (char *)plaintext};
    octet T = {0, 16, (char *)calculated_tag};
    if (!valid_gcm_input(key_len, iv_len, aad_len, ciphertext_len)) return 0;
    AES_GCM_DECRYPT(&K, &IV, &H, &C, &P, &T);
    if (P.len != ciphertext_len || T.len != 16 || !constant_time_equal(calculated_tag, expected_tag)) {
        memset(plaintext, 0, (size_t)ciphertext_len);
        return 0;
    }
    return 1;
}
