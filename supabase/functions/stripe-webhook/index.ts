try {
      const fullSession = await stripe.checkout.sessions.retrieve(session.id, {
        expand: ["line_items.data.price.product"],
      });
      productId = (fullSession.line_items?.data?.[0]?.price?.product as any)?.id || "";
      tierInfo = PRODUCT_TIER_MAP[productId] || PRODUCT_TIER_MAP["prod_UY4ePrftc3nhx1"];
      console.log(`Product ID: ${productId}, Tier: ${tierInfo.tier}`);
    } catch (err) {
      console.error("Could not retrieve session, defaulting to single_course:", err);
    }// Stripe webhook handler - renewal fix
import Stripe from "https://esm.sh/stripe@14.21.0?target=denonext";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2.39.0";

const stripe = new Stripe(Deno.env.get("STRIPE_SECRET_KEY")!, {
  apiVersion: "2024-06-20",
  httpClient: Stripe.createFetchHttpClient(),
});

const supabase = createClient(
  Deno.env.get("SUPABASE_URL")!,
  Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!
);

const webhookSecret = Deno.env.get("STRIPE_WEBHOOK_SECRET")!;

const PRODUCT_TIER_MAP: Record<string, { tier: string; max_instructors: number; max_students: number }> = {
  "prod_UY4ePrftc3nhx1": { tier: "single_course", max_instructors: 1, max_students: 40 },
  "prod_UY4ffFeS5Q98DP": { tier: "program_license", max_instructors: 5, max_students: 200 },
  "prod_UY4gVT3SsNO0fi": { tier: "institution", max_instructors: 9999, max_students: 9999 },
};

Deno.serve(async (req) => {
  const signature = req.headers.get("stripe-signature");
  if (!signature) return new Response("No signature", { status: 400 });

  const body = await req.text();
  let event: Stripe.Event;

  try {
    event = await stripe.webhooks.constructEventAsync(body, signature, webhookSecret);
  } catch (err) {
    console.error("Signature verification failed:", err);
    return new Response("Invalid signature", { status: 400 });
  }

  console.log(`Received event: ${event.type}`);

  if (event.type === "checkout.session.completed") {
    const session = event.data.object as Stripe.Checkout.Session;
    const email = session.customer_details?.email;
    const stripeCustomerId = session.customer as string;
    const stripeSubscriptionId = session.subscription as string;

    if (!email) return new Response("No customer email", { status: 400 });

    // Retrieve full session with line items expanded
    let productId = "";
    let tierInfo = PRODUCT_TIER_MAP["prod_UY4ePrftc3nhx1"];
    try {
      productId = session.metadata?.product_id || "";
      tierInfo = PRODUCT_TIER_MAP[productId] || PRODUCT_TIER_MAP["prod_UY4ePrftc3nhx1"];
      console.log(`Product ID from metadata: ${productId}, Tier: ${tierInfo.tier}`);
    } catch (err) {
      console.error("Could not read metadata, defaulting to single_course:", err);
    }

    const periodStart = new Date().toISOString();
    const periodEnd = new Date(Date.now() + 365 * 24 * 60 * 60 * 1000).toISOString();

    const { data: existingSub } = await supabase
      .from("subscriptions")
      .select("id, user_id")
      .eq("stripe_customer_id", stripeCustomerId)
      .single();

    let userId: string;

    if (existingSub) {
      userId = existingSub.user_id;
      console.log(`Renewal for ${email}`);

      await supabase.from("subscriptions").update({
        stripe_subscription_id: stripeSubscriptionId,
        stripe_product_id: productId,
        tier: tierInfo.tier,
        status: "active",
        current_period_start: periodStart,
        current_period_end: periodEnd,
      }).eq("stripe_customer_id", stripeCustomerId);

      await supabase.from("licenses").update({
        expires_at: periodEnd,
      }).eq("owner_user_id", userId);

      console.log(`Successfully renewed ${email} with ${tierInfo.tier}`);

    } else {
      const { data: inviteData, error: inviteError } = await supabase.auth.admin.inviteUserByEmail(email, {
        redirectTo: "https://app.epilabinteractive.com",
      });

      if (inviteError) {
        console.error("Invite error:", inviteError);
        return new Response(`Invite failed: ${inviteError.message}`, { status: 500 });
      }

      userId = inviteData.user.id;

      const { data: subData, error: subError } = await supabase.from("subscriptions").insert({
        user_id: userId,
        stripe_customer_id: stripeCustomerId,
        stripe_subscription_id: stripeSubscriptionId,
        stripe_product_id: productId,
        tier: tierInfo.tier,
        status: "active",
        current_period_start: periodStart,
        current_period_end: periodEnd,
      }).select().single();

      if (subError) {
        console.error("Subscription insert error:", subError);
        return new Response(`Subscription failed: ${subError.message}`, { status: 500 });
      }

      await supabase.from("licenses").insert({
        subscription_id: subData.id,
        owner_user_id: userId,
        max_instructors: tierInfo.max_instructors,
        max_students: tierInfo.max_students,
        expires_at: periodEnd,
      });

      await supabase.from("profiles").upsert({
        id: userId,
        role: "instructor",
        email: email,
      }, { onConflict: "id" });
      console.log(`Successfully provisioned ${email} with ${tierInfo.tier}`);
    }
  }

  return new Response(JSON.stringify({ received: true }), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
});