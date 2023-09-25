import torch
from dataset import nontumorousDataset
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from discriminator_model import Discriminator
from generator_model import Generator

def train_fn(
    disc_N, disc_T, gen_T, gen_N, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler, epoch, val_loader
):
    epoch_disc_loss = 0.0
    epoch_gen_loss = 0.0
    epoch_N_real = 0.0
    epoch_N_fake = 0.0
    epoch_T_real = 0.0
    epoch_T_fake = 0.0

    for tumorous, nontumorous in loader:
        tumorous = tumorous.to(config.DEVICE)
        nontumorous = nontumorous.to(config.DEVICE)

        # Train Discriminators H and Z
        with torch.cuda.amp.autocast():
            fake_nontumorous = gen_N(tumorous)
            D_N_real = disc_N(nontumorous)
            D_N_fake = disc_N(fake_nontumorous.detach())
            epoch_N_real += D_N_real.mean().item()
            epoch_N_fake += D_N_fake.mean().item()
            D_N_real_loss = mse(D_N_real, torch.ones_like(D_N_real))
            D_N_fake_loss = mse(D_N_fake, torch.zeros_like(D_N_fake))
            D_N_loss = D_N_real_loss + D_N_fake_loss

            fake_tumorous = gen_T(nontumorous)
            D_T_real = disc_T(tumorous)
            D_T_fake = disc_T(fake_tumorous.detach())
            epoch_T_real += D_T_real.mean().item()
            epoch_T_fake += D_T_fake.mean().item()
            D_T_real_loss = mse(D_T_real, torch.ones_like(D_T_real))
            D_T_fake_loss = mse(D_T_fake, torch.zeros_like(D_T_fake))
            D_T_loss = D_T_real_loss + D_T_fake_loss

            D_loss = (D_N_loss + D_T_loss) / 2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generators H and Z
        with torch.cuda.amp.autocast():
            D_N_fake = disc_N(fake_nontumorous)
            D_T_fake = disc_T(fake_tumorous)
            loss_G_N = mse(D_N_fake, torch.ones_like(D_N_fake))
            loss_G_T = mse(D_T_fake, torch.ones_like(D_T_fake))

            cycle_tumorous = gen_T(fake_nontumorous)
            cycle_nontumorous = gen_N(fake_tumorous)
            cycle_tumorous_loss = l1(tumorous, cycle_tumorous)
            cycle_nontumorous_loss = l1(nontumorous, cycle_nontumorous)

            G_loss = (
                loss_G_T
                + loss_G_N
                + cycle_tumorous_loss * config.LAMBDA_CYCLE
                + cycle_nontumorous_loss * config.LAMBDA_CYCLE
            )

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        epoch_disc_loss += D_loss.item()
        epoch_gen_loss += G_loss.item()

    # Wyświetl wartości strat i metryk na koniec epoki treningu
    print(f"Epoch [{epoch}/{config.NUM_EPOCHS}]")
    print(f"Discriminator Loss: {epoch_disc_loss:.4f}, Generator Loss: {epoch_gen_loss:.4f}")

    # Dodatkowo obliczamy straty walidacji podczas treningu
    validate_fn(disc_N, disc_T, gen_T, gen_N, val_loader, l1, mse, epoch)

def validate_fn(disc_N, disc_T, gen_T, gen_N, val_loader, l1, mse, epoch):
    disc_N.eval()
    disc_T.eval()
    gen_T.eval()
    gen_N.eval()

    epoch_N_real = 0.0
    epoch_N_fake = 0.0
    epoch_T_real = 0.0
    epoch_T_fake = 0.0

    with torch.no_grad():
        for tumorous, nontumorous in val_loader:
            tumorous = tumorous.to(config.DEVICE)
            nontumorous = nontumorous.to(config.DEVICE)

            # Validation for Discriminators H and Z
            fake_nontumorous = gen_N(tumorous)
            D_N_real = disc_N(nontumorous)
            D_N_fake = disc_N(fake_nontumorous)
            epoch_N_real += D_N_real.mean().item()
            epoch_N_fake += D_N_fake.mean().item()

            fake_tumorous = gen_T(nontumorous)
            D_T_real = disc_T(tumorous)
            D_T_fake = disc_T(fake_tumorous)
            epoch_T_real += D_T_real.mean().item()
            epoch_T_fake += D_T_fake.mean().item()

        num_samples = len(val_loader)
        avg_N_real = epoch_N_real / num_samples
        avg_N_fake = epoch_N_fake / num_samples
        avg_T_real = epoch_T_real / num_samples
        avg_T_fake = epoch_T_fake / num_samples

        print(f"Discriminator H Real: {avg_N_real:.4f}, Discriminator H Fake: {avg_N_fake:.4f}")
        print(f"Discriminator Z Real: {avg_T_real:.4f}, Discriminator Z Fake: {avg_T_fake:.4f}")

    disc_N.train()
    disc_T.train()
    gen_T.train()
    gen_N.train()

def main():
    disc_N = Discriminator(in_channels=3).to(config.DEVICE)
    disc_T = Discriminator(in_channels=3).to(config.DEVICE)
    gen_T = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_N = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    opt_disc = optim.Adam(
        list(disc_N.parameters()) + list(disc_T.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_T.parameters()) + list(gen_N.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN_N,
            gen_N,
            opt_gen,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_GEN_T,
            gen_T,
            opt_gen,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_N,
            disc_N,
            opt_disc,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_T,
            disc_T,
            opt_disc,
            config.LEARNING_RATE,
        )

    dataset = nontumorousDataset(
        root_nontumorous=config.TRAIN_DIR + "/nontumorous",
        root_tumorous=config.TRAIN_DIR + "/tumorous",
        transform=config.transforms,
    )
    val_dataset = nontumorousDataset(
        root_nontumorous="cyclegan_test/nontumorous",
        root_tumorous="cyclegan_test/tumorous",
        transform=config.transforms,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config.NUM_EPOCHS):
        train_fn(
            disc_N,
            disc_T,
            gen_T,
            gen_N,
            loader,
            opt_disc,
            opt_gen,
            L1,
            mse,
            d_scaler,
            g_scaler,
            epoch,
            val_loader
        )

        if config.SAVE_MODEL:
            save_checkpoint(gen_N, opt_gen, filename=config.CHECKPOINT_GEN_N)
            save_checkpoint(gen_T, opt_gen, filename=config.CHECKPOINT_GEN_T)
            save_checkpoint(disc_N, opt_disc, filename=config.CHECKPOINT_CRITIC_N)
            save_checkpoint(disc_T, opt_disc, filename=config.CHECKPOINT_CRITIC_T)

        # Po zakończeniu trenowania generatorów, zapisz ich wagi
        if config.SAVE_GENERATOR_MODEL:
            torch.save(gen_T.state_dict(), 'gen_T.pth')
            torch.save(gen_N.state_dict(), 'gen_N.pth')
