use stal_cli::cli::Cli;

fn main() -> anyhow::Result<()> {
    let env = env_logger::Env::default().default_filter_or("INFO");
    env_logger::builder().parse_env(env).init();

    Cli::execute()
}
