# Test-only strict counterpart of _n4me_compat.py and _n4me_compat.mjs.
# Retain every frame and learned byte; admit only the known writer ABI bump.
.n4me_uint <- function(bytes) {
  stopifnot(length(bytes) %in% c(4L, 8L))
  value <- sum(as.double(bytes) * 256^(seq_along(bytes) - 1L))
  stopifnot(is.finite(value), value >= 0, value <= 2^53 - 1)
  value
}

.n4me_fnv <- function(bytes) {
  # Two uint32 limbs keep all operations exactly below double's 2^53 limit.
  low <- 2216829733; high <- 3421674724
  for (byte in as.integer(bytes)) {
    low <- low - low %% 256 + bitwXor(as.integer(low %% 256), byte)
    product <- low * 435
    high <- (high * 435 + low * 256 + floor(product / 2^32)) %% 2^32
    low <- product %% 2^32
  }
  as.raw(c(floor(low / 256^(0:3)) %% 256, floor(high / 256^(0:3)) %% 256))
}

.n4me_checksum <- function(bytes) {
  stopifnot(is.raw(bytes), length(bytes) >= 28L,
            identical(tail(bytes, 8L), .n4me_fnv(head(bytes, -8L))))
  invisible(TRUE)
}

.n4me_model <- function(bytes, writer) {
  .n4me_checksum(bytes)
  stopifnot(identical(rawToChar(bytes[1:4]), "N4MM"))
  info <- n4m_model_inspect(bytes)
  stopifnot(identical(as.integer(info$writer_abi), writer))
  # The complete native decoder validates all dimensions, sections and checksum.
  descriptor <- n4m_model_descriptor(bytes)
  info$writer_abi <- c(0L, 0L, 0L)
  list(info = info, descriptor = descriptor, framing = bytes[1:8],
       learned = bytes[21:(length(bytes) - 8L)])
}

.n4me_decode <- function(bytes) {
  .n4me_checksum(bytes)
  stopifnot(identical(rawToChar(bytes[1:4]), "N4ME"), .n4me_uint(bytes[5:8]) == 1)
  writer <- as.integer(vapply(c(9L, 13L, 17L), function(i) .n4me_uint(bytes[i:(i + 3L)]), numeric(1)))
  offset <- 21L
  take <- function(size) {
    stopifnot(is.finite(size), size >= 0, size == floor(size), size <= length(bytes) - 8L - offset + 1L)
    if (!size) return(raw())
    value <- bytes[offset:(offset + size - 1L)]
    offset <<- offset + size
    value
  }
  integer <- function(width) .n4me_uint(take(width))
  text <- function() take(integer(4L))
  method <- text()
  parameters <- lapply(seq_len(integer(4L)), function(i) {
    name <- text(); kind <- integer(4L); count <- integer(8L)
    list(name, kind, count, take(count * 8))
  })
  capabilities_and_dimensions <- take(24L)
  blocks <- lapply(seq_len(integer(4L)), function(i) {
    tag <- integer(4L); block <- take(integer(8L))
    if (tag == 1296905294) {
      return(list(tag, .n4me_model(block, writer)))
    }
    if (tag == 827542595 && rawToChar(method) %in% c(
      "models.classification.pls_lda", "models.classification.pls_logistic",
      "models.sparse.sparse_pls_da")) {
      stopifnot(length(block) >= 16L)
      classes <- .n4me_uint(block[9:16])
      stopifnot(classes >= 2, classes <= 2^20)
      frame <- 17 + classes * 8
      stopifnot(frame + 15 <= length(block))
      size <- .n4me_uint(block[frame:(frame + 7)])
      packed <- .n4me_uint(block[(frame + 8):(frame + 15)])
      start <- frame + 16; end <- start + size - 1
      stopifnot(size >= 28, size <= 2^30, packed == ceiling(size / 8),
                start + packed * 8 - 1 <= length(block))
      suffix <- if (end < length(block)) block[(end + 1):length(block)] else raw()
      return(list(tag, block[1:(start - 1)], .n4me_model(block[start:end], writer), suffix))
    }
    list(tag, block)
  })
  stopifnot(offset == length(bytes) - 7L)
  list(writer = writer, framing = bytes[1:8], state = list(method, parameters, capabilities_and_dimensions, blocks))
}

assert_n4me_reexport_equivalent <- function(original, current) {
  before <- .n4me_decode(original); after <- .n4me_decode(current)
  stopifnot(identical(before$writer, c(2L, 15L, 0L)),
            identical(after$writer, as.integer(n4m_abi_version())),
            identical(before$framing, after$framing), identical(before$state, after$state))
  old_native <- n4m_estimator_import(original)
  new_native <- n4m_estimator_import(current)
  stopifnot(identical(class(old_native), class(new_native)),
            identical(old_native$method_id, new_native$method_id),
            identical(old_native$params, new_native$params))
  invisible(TRUE)
}
