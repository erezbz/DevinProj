// Deterministic headless GBA runner (libmgba). Usage: runner ROM OUTPREFIX "script tokens..."
// Tokens: wN (idle N frames), A/B/START/SELECT/UP/DOWN/LEFT/RIGHT/L/R (tap), K1+K2 (chord tap),
//         holdK:N (hold N frames), shot (PNG), X*N (repeat), save:FILE, load:FILE
#include <mgba/core/core.h>
#include <mgba/core/config.h>
#include <mgba/core/serialize.h>
#include <mgba-util/vfs.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>

static struct mCore *core;
static color_t *buf;
static unsigned W, H;
static const char *prefix;
static int shotN;

static int keybit(const char *k) {
    static const char *names[] = {"A","B","SELECT","START","RIGHT","LEFT","UP","DOWN","R","L"};
    for (int i = 0; i < 10; i++) if (!strcmp(k, names[i])) return 1 << i;
    fprintf(stderr, "bad key %s\n", k); exit(2);
}
static void frames(int keys, int n) { core->setKeys(core, keys); for (int i = 0; i < n; i++) core->runFrame(core); }
static void shot(void) {
    char path[512]; snprintf(path, sizeof path, "%s_%02d.png", prefix, shotN++);
    FILE *f = fopen(path, "wb");
    png_structp p = png_create_write_struct(PNG_LIBPNG_VER_STRING, 0, 0, 0);
    png_infop info = png_create_info_struct(p);
    png_init_io(p, f);
    png_set_IHDR(p, info, W, H, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_write_info(p, info);
    unsigned char *row = malloc(W * 3);
    for (unsigned y = 0; y < H; y++) {
        for (unsigned x = 0; x < W; x++) {
            uint32_t c = (uint32_t)buf[y * W + x];
            if (sizeof(color_t) == 4) { row[x*3] = c & 0xFF; row[x*3+1] = (c >> 8) & 0xFF; row[x*3+2] = (c >> 16) & 0xFF; }
            else { row[x*3] = (c & 0x1F) << 3; row[x*3+1] = ((c >> 5) & 0x3F) << 2; row[x*3+2] = ((c >> 11) & 0x1F) << 3; }
        }
        png_write_row(p, row);
    }
    png_write_end(p, info); png_destroy_write_struct(&p, &info); fclose(f); free(row);
    printf("%s\n", path);
}
static void token(char *t) {
    char *star = strchr(t, '*');
    if (star) { int n = atoi(star + 1); *star = 0; for (int i = 0; i < n; i++) { char c[64]; strcpy(c, t); token(c); } return; }
    if (!strcmp(t, "shot")) shot();
    else if (!strncmp(t, "rec:", 4)) { int n, every; sscanf(t + 4, "%d:%d", &n, &every); for (int i = 0; i < n; i += every) { frames(0, every); shot(); } }
    else if (t[0] == 'w' && t[1] >= '0' && t[1] <= '9') frames(0, atoi(t + 1));
    else if (!strncmp(t, "hold", 4)) { char *c = strchr(t, ':'); *c = 0; frames(keybit(t + 4), atoi(c + 1)); frames(0, 2); }
    else if (!strncmp(t, "save:", 5)) { struct VFile *vf = VFileOpen(t + 5, O_CREAT | O_TRUNC | O_RDWR); mCoreSaveStateNamed(core, vf, SAVESTATE_SAVEDATA | SAVESTATE_RTC); vf->close(vf); }
    else if (!strncmp(t, "load:", 5)) { struct VFile *vf = VFileOpen(t + 5, O_RDONLY); mCoreLoadStateNamed(core, vf, SAVESTATE_SAVEDATA | SAVESTATE_RTC); vf->close(vf); }
    else { int keys = 0; for (char *k = strtok(t, "+"); k; k = strtok(NULL, "+")) keys |= keybit(k); frames(keys, 4); frames(0, 4); }
}
int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage\n"); return 1; }
    prefix = argv[2];
    core = mCoreFind(argv[1]);
    core->init(core);
    mCoreInitConfig(core, NULL);
    core->desiredVideoDimensions(core, &W, &H);
    buf = malloc(W * H * sizeof(color_t));
    core->setVideoBuffer(core, buf, W);
    if (!mCoreLoadFile(core, argv[1])) { fprintf(stderr, "load fail\n"); return 1; }
    core->reset(core);
    char *script = strdup(argv[3]);
    char *save; char *t = strtok_r(script, " \n", &save);
    while (t) { char tmp[256]; strncpy(tmp, t, 255); tmp[255] = 0; token(tmp); t = strtok_r(NULL, " \n", &save); }
    core->deinit(core);
    return 0;
}
