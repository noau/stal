fn main() {
    let env = env_logger::Env::default().default_filter_or("INFO");
    env_logger::builder().parse_env(env).init();
}
