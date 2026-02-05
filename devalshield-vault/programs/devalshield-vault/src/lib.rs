use anchor_lang::prelude::*;
use anchor_lang::solana_program::program::invoke;
use anchor_spl::token::{self, Mint, Token, TokenAccount, Transfer, Burn, MintTo};
use mpl_core::{ID as MPL_CORE_ID, instructions::CreateV2CpiBuilder};

declare_id!("BppKFtdQqyLtn6PZtME52UrXDUyi67Lwn68UaXBUSrtc");

#[program]
pub mod devalshield_vault {
    use super::*;

    pub fn initialize(ctx: Context<InitializeVault>, performance_fee: u16, management_fee: u16) -> Result<()> {
        ctx.accounts.vault.set_inner(Vault {
            authority: ctx.accounts.authority.key(),
            usdt_vault: ctx.accounts.usdt_vault.key(),
            usdc_payout: ctx.accounts.usdc_payout.key(),
            shares_mint: ctx.accounts.shares_mint.key(),
            index_oracle: ctx.accounts.index_oracle.key(),
            treasury: ctx.accounts.treasury.key(),
            triggered: false,
            performance_fee_bps: performance_fee,
            management_fee_bps: management_fee,
            bump: ctx.bumps.vault,
        });
        Ok(())
    }

    pub fn stake(ctx: Context<Stake>, amount: u64, name: String, uri: String) -> Result<()> {
        let vault = &ctx.accounts.vault;
        
        // 1. Take management fee (0.5% default)
        let mgmt_fee = (amount as u128 * vault.management_fee_bps as u128 / 10_000) as u64;
        let net_amount = amount - mgmt_fee;

        // Transfer mgmt_fee to treasury
        let cpi_fee = CpiContext::new(
            ctx.accounts.token_program.to_account_info(),
            Transfer {
                from: ctx.accounts.user_usdt.to_account_info(),
                to: ctx.accounts.treasury_usdt.to_account_info(),
                authority: ctx.accounts.authority.to_account_info(),
            },
        );
        token::transfer(cpi_fee, mgmt_fee)?;

        // 2. Transfer net USDT to vault
        let cpi_ctx = CpiContext::new(
            ctx.accounts.token_program.to_account_info(),
            Transfer {
                from: ctx.accounts.user_usdt.to_account_info(),
                to: ctx.accounts.usdt_vault.to_account_info(),
                authority: ctx.accounts.authority.to_account_info(),
            },
        );
        token::transfer(cpi_ctx, net_amount)?;

        // 3. Mint shares (1:1 with net amount)
        let seeds = &[b"vault" as &[u8], &[vault.bump]];
        let signer = &[&seeds[..]];
        let cpi_mint = CpiContext::new_with_signer(
            ctx.accounts.token_program.to_account_info(),
            MintTo {
                mint: ctx.accounts.shares_mint.to_account_info(),
                to: ctx.accounts.user_shares.to_account_info(),
                authority: ctx.accounts.vault_authority.to_account_info(),
            },
            signer,
        );
        token::mint_to(cpi_mint, net_amount)?;

        // 4. Mint Shield NFT badge
        CreateV2CpiBuilder::new(&ctx.accounts.mpl_core.to_account_info())
            .asset(&ctx.accounts.shield_nft.to_account_info())
            .payer(&ctx.accounts.authority.to_account_info())
            .owner(Some(&ctx.accounts.authority.to_account_info()))
            .name(name)
            .uri(uri)
            .invoke()?;

        Ok(())
    }

    pub fn rebalance_low_index(ctx: Context<RebalanceLowIndex>, amount: u64) -> Result<()> {
        let vault = &ctx.accounts.vault;
        require!(!vault.triggered, ErrorCode::AlreadyTriggered);

        // Placeholder for Jupiter Swap + Jito Deposit
        // In real implementation, we would use Jupiter CPI to swap USDT -> SOL
        // then spl_stake_pool::instruction::deposit_sol to get JitoSOL
        
        msg!("Rebalanced {} USDT to Yield Strategy (JitoSOL)", amount);
        Ok(())
    }

    pub fn trigger_unwind(ctx: Context<TriggerUnwind>) -> Result<()> {
        let vault = &mut ctx.accounts.vault;
        require!(!vault.triggered, ErrorCode::AlreadyTriggered);

        // Simple index check (MVP)
        let index: u64 = ctx.accounts.index_oracle.data.borrow()[0] as u64; 
        require!(index > 75, ErrorCode::IndexTooLow);

        let amount = ctx.accounts.usdt_vault.amount / 2;
        let seeds = &[b"vault" as &[u8], &[vault.bump]];
        let signer = &[&seeds[..]];

        let cpi_ctx = CpiContext::new_with_signer(
            ctx.accounts.token_program.to_account_info(),
            Transfer {
                from: ctx.accounts.usdt_vault.to_account_info(),
                to: ctx.accounts.usdc_payout.to_account_info(),
                authority: ctx.accounts.vault_authority.to_account_info(),
            },
            signer,
        );
        token::transfer(cpi_ctx, amount)?;

        vault.triggered = true;
        Ok(())
    }

    pub fn claim(ctx: Context<Claim>) -> Result<()> {
        let vault = &ctx.accounts.vault;
        require!(vault.triggered, ErrorCode::NotTriggered);

        let shares = ctx.accounts.user_shares.amount;
        let total_shares = ctx.accounts.shares_mint.supply;
        require!(total_shares > 0, ErrorCode::NoSharesMinted);
        
        let gross_payout = (shares as u128 * ctx.accounts.usdc_payout.amount as u128 / total_shares as u128) as u64;
        
        // Performance fee (10% default)
        let perf_fee = (gross_payout as u128 * vault.performance_fee_bps as u128 / 10_000) as u64;
        let net_payout = gross_payout - perf_fee;

        let seeds = &[b"vault" as &[u8], &[vault.bump]];
        let signer = &[&seeds[..]];

        // 1. Fee to Treasury
        token::transfer(
            CpiContext::new_with_signer(
                ctx.accounts.token_program.to_account_info(),
                Transfer {
                    from: ctx.accounts.usdc_payout.to_account_info(),
                    to: ctx.accounts.treasury_usdc.to_account_info(),
                    authority: ctx.accounts.vault_authority.to_account_info(),
                },
                signer,
            ),
            perf_fee
        )?;

        // 2. Net payout to user
        token::transfer(
            CpiContext::new_with_signer(
                ctx.accounts.token_program.to_account_info(),
                Transfer {
                    from: ctx.accounts.usdc_payout.to_account_info(),
                    to: ctx.accounts.user_usdc.to_account_info(),
                    authority: ctx.accounts.vault_authority.to_account_info(),
                },
                signer,
            ),
            net_payout
        )?;

        // 3. Burn shares
        token::burn(
            CpiContext::new(
                ctx.accounts.token_program.to_account_info(),
                Burn {
                    mint: ctx.accounts.shares_mint.to_account_info(),
                    from: ctx.accounts.user_shares.to_account_info(),
                    authority: ctx.accounts.authority.to_account_info(),
                },
            ),
            shares
        )?;

        Ok(())
    }
}

#[account]
pub struct Vault {
    pub authority: Pubkey,
    pub usdt_vault: Pubkey,
    pub usdc_payout: Pubkey,
    pub shares_mint: Pubkey,
    pub index_oracle: Pubkey,
    pub treasury: Pubkey,
    pub triggered: bool,
    pub performance_fee_bps: u16,
    pub management_fee_bps: u16,
    pub bump: u8,
}

#[derive(Accounts)]
pub struct InitializeVault<'info> {
    #[account(init, payer = authority, space = 8 + 32*6 + 1 + 2*2 + 1, seeds = [b"vault"], bump)]
    pub vault: Account<'info, Vault>,
    #[account(mut)]
    pub usdt_vault: Account<'info, TokenAccount>,
    #[account(mut)]
    pub usdc_payout: Account<'info, TokenAccount>,
    pub shares_mint: Account<'info, Mint>,
    /// CHECK: Simple index oracle PDA
    pub index_oracle: AccountInfo<'info>,
    /// CHECK: Treasury account
    pub treasury: AccountInfo<'info>,
    #[account(mut)]
    pub authority: Signer<'info>,
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct Stake<'info> {
    #[account(mut, seeds = [b"vault"], bump = vault.bump)]
    pub vault: Account<'info, Vault>,
    #[account(mut)]
    pub user_usdt: Account<'info, TokenAccount>,
    #[account(mut, address = vault.usdt_vault)]
    pub usdt_vault: Account<'info, TokenAccount>,
    #[account(mut, address = vault.treasury)]
    pub treasury_usdt: Account<'info, TokenAccount>,
    #[account(mut)]
    pub user_shares: Account<'info, TokenAccount>,
    #[account(mut, address = vault.shares_mint)]
    pub shares_mint: Account<'info, Mint>,
    /// CHECK: PDA
    #[account(seeds = [b"vault_authority"], bump)]
    pub vault_authority: UncheckedAccount<'info>,
    /// CHECK: Shield NFT account
    #[account(mut)]
    pub shield_nft: AccountInfo<'info>,
    pub authority: Signer<'info>,
    pub token_program: Program<'info, Token>,
    /// CHECK: MPL Core program
    pub mpl_core: AccountInfo<'info>,
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct RebalanceLowIndex<'info> {
    #[account(mut, seeds = [b"vault"], bump = vault.bump)]
    pub vault: Account<'info, Vault>,
    #[account(mut, address = vault.usdt_vault)]
    pub usdt_vault: Account<'info, TokenAccount>,
    /// CHECK: Jito Stake Pool
    pub jito_stake_pool: AccountInfo<'info>,
    pub token_program: Program<'info, Token>,
}

#[derive(Accounts)]
pub struct TriggerUnwind<'info> {
    #[account(mut, seeds = [b"vault"], bump = vault.bump)]
    pub vault: Account<'info, Vault>,
    #[account(mut, address = vault.usdt_vault)]
    pub usdt_vault: Account<'info, TokenAccount>,
    #[account(mut, address = vault.usdc_payout)]
    pub usdc_payout: Account<'info, TokenAccount>,
    /// CHECK: PDA
    #[account(seeds = [b"vault_authority"], bump)]
    pub vault_authority: UncheckedAccount<'info>,
    /// CHECK: Simple index oracle PDA
    pub index_oracle: AccountInfo<'info>,
    pub authority: Signer<'info>,
    pub token_program: Program<'info, Token>,
}

#[derive(Accounts)]
pub struct Claim<'info> {
    #[account(mut, seeds = [b"vault"], bump = vault.bump)]
    pub vault: Account<'info, Vault>,
    #[account(mut)]
    pub user_shares: Account<'info, TokenAccount>,
    #[account(mut, address = vault.shares_mint)]
    pub shares_mint: Account<'info, Mint>,
    #[account(mut, address = vault.usdc_payout)]
    pub usdc_payout: Account<'info, TokenAccount>,
    #[account(mut)]
    pub user_usdc: Account<'info, TokenAccount>,
    #[account(mut, address = vault.treasury)]
    pub treasury_usdc: Account<'info, TokenAccount>,
    /// CHECK: PDA
    #[account(seeds = [b"vault_authority"], bump)]
    pub vault_authority: UncheckedAccount<'info>,
    pub authority: Signer<'info>,
    pub token_program: Program<'info, Token>,
}

#[error_code]
pub enum ErrorCode {
    #[msg("Index too low to trigger")]
    IndexTooLow,
    #[msg("Already triggered")]
    AlreadyTriggered,
    #[msg("Not triggered yet")]
    NotTriggered,
    #[msg("No shares minted")]
    NoSharesMinted,
}
