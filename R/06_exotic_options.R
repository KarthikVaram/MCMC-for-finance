# Stage 6: Exotic Option Pricing

library(Rcpp)
library(ggplot2)

sourceCpp("src/exotic_options.cpp")
sourceCpp("src/mc_pricer.cpp")

# ── Parameters ───────────────────────────────────────────────────────────────
S0 <- 100; K <- 100; r <- 0.05; sigma <- 0.20; T <- 1.0
N  <- 100000; steps <- 252; seed <- 42

# ── 1. Asian Options ─────────────────────────────────────────────────────────
cat("\n=== Asian Options ===\n")

asian_c <- asian_call(S0, K, r, sigma, T, N, steps, seed)
asian_p <- asian_put(S0, K, r, sigma, T, N, steps, seed)

euro_c  <- mc_call(S0, K, r, sigma, T, N, steps, seed)
euro_p  <- mc_put(S0, K, r, sigma, T, N, steps, seed)

cat(sprintf("Asian Call  price: %.4f  (SE: %.5f)\n", asian_c$price, asian_c$std_error))
cat(sprintf("Euro  Call  price: %.4f  (SE: %.5f)\n", euro_c$price,  euro_c$std_error))
cat(sprintf("Asian Put   price: %.4f  (SE: %.5f)\n", asian_p$price, asian_p$std_error))
cat(sprintf("Euro  Put   price: %.4f  (SE: %.5f)\n", euro_p$price,  euro_p$std_error))

# ── 2. Barrier Options ───────────────────────────────────────────────────────
cat("\n=== Barrier Options ===\n")

H_down <- 80   # lower barrier
H_up   <- 120  # upper barrier

do_call <- barrier_down_out_call(S0, K, H_down, r, sigma, T, N, steps, seed)
uo_call <- barrier_up_out_call(S0, K, H_up,   r, sigma, T, N, steps, seed)

cat(sprintf("Down-and-Out Call (H=%.0f): %.4f  (%.1f%% knocked out)\n",
            H_down, do_call$price, do_call$pct_knocked_out))
cat(sprintf("Up-and-Out   Call (H=%.0f): %.4f  (%.1f%% knocked out)\n",
            H_up,   uo_call$price, uo_call$pct_knocked_out))
cat(sprintf("Vanilla European Call:  %.4f\n", euro_c$price))

# Barrier level sensitivity
barriers_down <- seq(60, 98, by = 4)
do_prices <- sapply(barriers_down, function(H) {
  barrier_down_out_call(S0, K, H, r, sigma, T, N, steps, seed)$price
})

df_barrier <- data.frame(barrier = barriers_down, price = do_prices)
p_barrier  <- ggplot(df_barrier, aes(barrier, price)) +
  geom_line(colour = "#e6550d", linewidth = 1.1) +
  geom_point(colour = "#e6550d", size = 2.5) +
  geom_hline(yintercept = euro_c$price, linetype = "dashed", colour = "grey40") +
  labs(title = "Down-and-Out Call: Price vs Lower Barrier",
       x = "Barrier Level H", y = "Option Price",
       caption = "Dashed = Vanilla European Call") +
  theme_minimal()
print(p_barrier)

# ── 3. Lookback Options ──────────────────────────────────────────────────────
cat("\n=== Lookback Options ===\n")

lb_call <- lookback_call(S0, r, sigma, T, N, steps, seed)
lb_put  <- lookback_put(S0,  r, sigma, T, N, steps, seed)

cat(sprintf("Lookback Call (S_T - min S): %.4f  (SE: %.5f)\n", lb_call$price, lb_call$std_error))
cat(sprintf("Lookback Put  (max S - S_T): %.4f  (SE: %.5f)\n", lb_put$price,  lb_put$std_error))

# ── Summary comparison ───────────────────────────────────────────────────────
summary_df <- data.frame(
  Product = c("European Call", "Asian Call", "Down-Out Call", "Up-Out Call", "Lookback Call"),
  Price   = c(euro_c$price, asian_c$price, do_call$price, uo_call$price, lb_call$price)
)
cat("\n=== Summary ===\n")
print(summary_df)

p_summary <- ggplot(summary_df, aes(reorder(Product, Price), Price, fill = Product)) +
  geom_col(show.legend = FALSE, width = 0.6) +
  coord_flip() +
  scale_fill_brewer(palette = "Set3") +
  labs(title = "Exotic Option Price Comparison",
       x = NULL, y = "Option Price") +
  theme_minimal()
print(p_summary)
