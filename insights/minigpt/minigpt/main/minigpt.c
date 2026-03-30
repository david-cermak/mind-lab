#include <stdio.h>
#include "esp_console.h"
#include "argtable3/argtable3.h"
#include "esp_log.h"
#include "esp_event.h"

static esp_console_repl_t *s_repl = NULL;
void infer(const char *prefix);

static struct {
    struct arg_str *prefix;
    struct arg_end *end;
} infer_args;

static int do_infer_cmd(int argc, char **argv)
{
    int nerrors = arg_parse(argc, argv, (void **)&infer_args);
    if (nerrors != 0) {
        arg_print_errors(stderr, infer_args.end, argv[0]);
        return 1;
    }
    infer(infer_args.prefix->sval[0]);
    return 0;
}

static void register_infer(void)
{
    infer_args.prefix = arg_str1(NULL, NULL, "<prefix>", "Prefix");
    infer_args.end = arg_end(1);
    const esp_console_cmd_t infer_cmd = {
        .command = "infer",
        .help = "infer",
        .hint = NULL,
        .func = &do_infer_cmd,
        .argtable = &infer_args
    };
    ESP_ERROR_CHECK(esp_console_cmd_register(&infer_cmd));
}

void app_main(void)
{
    ESP_ERROR_CHECK(esp_event_loop_create_default());

    esp_console_repl_config_t repl_config = ESP_CONSOLE_REPL_CONFIG_DEFAULT();
    // install console REPL environment
    esp_console_dev_uart_config_t uart_config = ESP_CONSOLE_DEV_UART_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_console_new_repl_uart(&uart_config, &repl_config, &s_repl));

    register_infer();
    ESP_ERROR_CHECK(esp_console_start_repl(s_repl));
}