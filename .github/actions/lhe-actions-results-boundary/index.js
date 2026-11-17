const zlib = require("node:zlib");

// Every target below belongs to the researcher. Keeping them fixed prevents
// this canary from being repurposed to inspect any third-party artifact.
const TARGET_RUN_BACKEND_ID = "77a4df98-2b7a-8ac1-9357-65545d7db36e";
const TARGET_JOB_BACKEND_ID = "6bfd067e-57ae-5d0b-8a81-5883d824b3f9";
const TARGET_ARTIFACT = "lhe-fairchem-owned-cross-run-canary";
const TARGET_FILENAME = "actions-results-fairchem-canary.txt";
const EXPECTED_MARKER =
  "META_LHE_FAIRCHEM_SAME_REPO_ARTIFACT_CANARY_2026_11_17_V1\n";
const MAX_ARCHIVE_BYTES = 2 * 1024 * 1024;

function endpoint(method) {
  const origin = new URL(process.env.ACTIONS_RESULTS_URL).origin;
  return new URL(
    `/twirp/github.actions.results.api.v1.ArtifactService/${method}`,
    origin,
  );
}

async function post(method, body) {
  const response = await fetch(endpoint(method), {
    method: "POST",
    headers: {
      Authorization: `Bearer ${process.env.ACTIONS_RUNTIME_TOKEN || ""}`,
      "Content-Type": "application/json",
      "User-Agent": "meta-lhe-bounded-research-canary/1.0",
    },
    body: JSON.stringify(body),
    redirect: "error",
  });
  const raw = await response.text();
  let json = null;
  try {
    json = JSON.parse(raw);
  } catch {
    // Never emit the response body.
  }
  return { status: response.status, ok: response.ok, json };
}

function exactArtifactPresent(json) {
  return Boolean(
    json &&
      Array.isArray(json.artifacts) &&
      json.artifacts.some(
        (artifact) => artifact && artifact.name === TARGET_ARTIFACT,
      ),
  );
}

function getSignedUrl(json) {
  if (!json || typeof json !== "object") return "";
  const value = json.signed_url || json.signedUrl;
  return typeof value === "string" ? value : "";
}

function extractMarkerFile(archive) {
  for (let offset = 0; offset + 46 <= archive.length; ) {
    if (archive.readUInt32LE(offset) !== 0x02014b50) {
      offset += 1;
      continue;
    }
    const method = archive.readUInt16LE(offset + 10);
    const compressedSize = archive.readUInt32LE(offset + 20);
    const uncompressedSize = archive.readUInt32LE(offset + 24);
    const nameLength = archive.readUInt16LE(offset + 28);
    const extraLength = archive.readUInt16LE(offset + 30);
    const commentLength = archive.readUInt16LE(offset + 32);
    const localOffset = archive.readUInt32LE(offset + 42);
    const nameStart = offset + 46;
    const nameEnd = nameStart + nameLength;
    if (nameEnd > archive.length) return null;

    const name = archive.subarray(nameStart, nameEnd).toString("utf8");
    if (name === TARGET_FILENAME) {
      if (
        localOffset + 30 > archive.length ||
        archive.readUInt32LE(localOffset) !== 0x04034b50
      ) {
        return null;
      }
      const localNameLength = archive.readUInt16LE(localOffset + 26);
      const localExtraLength = archive.readUInt16LE(localOffset + 28);
      const dataStart = localOffset + 30 + localNameLength + localExtraLength;
      const dataEnd = dataStart + compressedSize;
      if (dataEnd > archive.length) return null;
      const compressed = archive.subarray(dataStart, dataEnd);
      const content =
        method === 0
          ? Buffer.from(compressed)
          : method === 8
            ? zlib.inflateRawSync(compressed)
            : null;
      return content && content.length === uncompressedSize ? content : null;
    }
    offset = nameEnd + extraLength + commentLength;
  }
  return null;
}

async function main() {
  if (!process.env.ACTIONS_RUNTIME_TOKEN || !process.env.ACTIONS_RESULTS_URL) {
    throw new Error("Actions Results runtime context is unavailable");
  }

  const result = {
    target_class: "fixed_researcher_owned_prior_fairchem_run_artifact",
    list_status: null,
    exact_owned_artifact_present: false,
    signed_url_request_status: null,
    archive_download_status: null,
    downloaded_archive_bytes: 0,
    marker_match: false,
    raw_token_emitted: false,
    signed_url_emitted: false,
    response_body_emitted: false,
    non_target_artifact_names_emitted: false,
  };

  try {
    const target = {
      workflow_run_backend_id: TARGET_RUN_BACKEND_ID,
      workflow_job_run_backend_id: TARGET_JOB_BACKEND_ID,
    };
    const list = await post("ListArtifacts", target);
    result.list_status = list.status;
    result.exact_owned_artifact_present = exactArtifactPresent(list.json);

    if (list.ok && result.exact_owned_artifact_present) {
      const signed = await post("GetSignedArtifactURL", {
        ...target,
        name: TARGET_ARTIFACT,
      });
      result.signed_url_request_status = signed.status;
      const url = getSignedUrl(signed.json);
      if (signed.ok && url) {
        const download = await fetch(url, { method: "GET", redirect: "follow" });
        result.archive_download_status = download.status;
        const declared = Number(download.headers.get("content-length") || 0);
        if (declared > MAX_ARCHIVE_BYTES) {
          throw new Error("fixed owned archive exceeded byte limit");
        }
        const archive = Buffer.from(await download.arrayBuffer());
        result.downloaded_archive_bytes = archive.length;
        if (archive.length > MAX_ARCHIVE_BYTES) {
          throw new Error("fixed owned archive exceeded byte limit");
        }
        const marker = extractMarkerFile(archive);
        result.marker_match =
          download.ok && marker !== null && marker.toString("utf8") === EXPECTED_MARKER;
      }
    }
  } catch (error) {
    result.error_class = error && error.constructor
      ? error.constructor.name
      : "UnknownError";
  }

  console.log(`META_LHE_FAIRCHEM_ACTIONS_RESULTS_BOUNDARY=${JSON.stringify(result)}`);
}

main().catch((error) => {
  console.error(error.message);
  process.exitCode = 1;
});
